import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, override

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import noise_cancellation, openai, silero
from livekit.plugins.openai.realtime.realtime_model import InputAudioTranscription
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from utils import ChatCtxDumper, get_datetime_str, load_chat_ctx_from_file

logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv()


CHAT_CTX_DUMP_FOLDER = f"chat_ctx_dump/{get_datetime_str(use_jst=True)}"


@dataclass
class UserData:
    """Stores data and agents to be shared across roleplay sections."""

    ctx: JobContext
    curr_section: str


def get_instruction(interview_language: str) -> str:
    system_prompt = "".join(
        [
            "You are Naomi, a 22-year-old, final-year Japanese university student"
            + "who is currently looking for a job. ",
            "You have no prior work experience and are not used to interviews. ",
            "Currently, you are interviewing with a hiring manager from a company you applied to, ",
            "and your goal is to get an offer.",
            "\n\n",
            "The company you applied is an IT consulting company specialized in AI products. ",
            "The job position is an entry-level AI engineer. ",
            "In general, you are expected to answer questions related to your background, "
            + "research, hobbies, and job hunting status. ",
            "The interviewer might also ask you detailed technical and behavioral questions "
            + "to understand your thinking process and motivation. ",
            "\n\n",
            "Remember, you should behave as a student who is not used to interviews. ",
            "Craft your response so that it befits your age . ",
            f"The interview is conducted in {interview_language}.",
        ]
    )
    return system_prompt


def session_context_manager(
    session: AgentSession[UserData], max_items: int = 8, ctx_trim_ratio=0.7
) -> None:
    """Truncate the context if it is too long.
    Only keep system prompt and user/agent messages.
    """
    if len(session._agent.chat_ctx.items) > max_items:
        new_chat_ctx = session._agent.chat_ctx.copy()
        background_tasks = set()

        logger.info(f"Chat context size: {len(new_chat_ctx.items)}, truncating...")
        max_itms = round(len(new_chat_ctx.items) * (1.0 - ctx_trim_ratio))
        if max_itms <= 0:
            max_itms = 1
        new_chat_ctx = new_chat_ctx.truncate(max_items=max_itms)
        logger.info(f"Start update chat context, size: {len(new_chat_ctx.items)}")
        task = asyncio.create_task(session._agent.update_chat_ctx(new_chat_ctx))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)


class BaseStudentAgent(Agent):
    def __init__(
        self,
        ctx: JobContext,
        interview_lang: Literal["en", "ja"] = "en",
        seed_chat_ctx_filepath: str | None = None,
        chat_ctx: ChatContext | None = None,
        ctx_max_n: int = 25,
        ctx_trim_ratio: float = 0.7,
    ) -> None:
        """Constructor.

        :param JobContext ctx: job context
        :param Literal["en", "ja"] interview_lang: interview language, defaults to "en"
        :param str | None seed_chat_ctx_filepath: to populate initial chat context, defaults to None
        :param str | None chat_ctx: initial chat context, defaults to None
        :param int ctx_max_n: max items in chat_ctx, defaults to 25
        :param float ctx_trim_ratio: how many items to truncate, defaults to 0.7
        """
        # conversation language
        inst_lang = "English" if interview_lang == "en" else "Japanese"
        stt_lang = "en" if interview_lang == "en" else "ja"

        # continue from previous conversation
        if seed_chat_ctx_filepath is not None:
            chat_ctx = load_chat_ctx_from_file(seed_chat_ctx_filepath, add_fake_messages=False)
        # else use the chat_ctx provided in constructor

        # initialize the agent
        super().__init__(
            instructions=get_instruction(inst_lang),
            chat_ctx=chat_ctx,
            allow_interruptions=True,
            turn_detection=MultilingualModel(unlikely_threshold=0.7),
            vad=ctx.proc.userdata["vad"],
            stt=openai.STT(language=stt_lang, model="gpt-4o-mini-transcribe"),
            llm=openai.realtime.RealtimeModel(
                model="gpt-4o-realtime-preview-2024-12-17",
                voice="sage",
                turn_detection=None,
                input_audio_transcription=InputAudioTranscription(
                    language=stt_lang, model="gpt-4o-mini-transcribe"
                ),
            ),
        )

        # other stuffs
        self.interview_lang = interview_lang
        self.ctx_max_n = ctx_max_n
        self.ctx_trim_ratio = ctx_trim_ratio

        # background asyncio tasks
        self.background_tasks: set[asyncio.Task] = set()

    async def on_enter(self) -> None:
        if self.chat_ctx is not None:
            self.session.generate_reply()

    def context_management(self) -> None:
        """Truncate the context if it is too long.
        Only keep system prompt and user/agent messages.
        """
        if self.chat_ctx is not None:
            new_chat_ctx = self.chat_ctx.copy()

            if len(new_chat_ctx.items) > self.ctx_max_n:
                logger.info(f"Chat context size: {len(new_chat_ctx.items)}, truncating...")
                max_itms = round(len(new_chat_ctx.items) * (1.0 - self.ctx_trim_ratio))
                if max_itms <= 0:
                    max_itms = 1
                new_chat_ctx = new_chat_ctx.truncate(max_items=max_itms)
                logger.info(f"Start update chat context, size: {len(new_chat_ctx.items)}")
                task = asyncio.create_task(self.update_chat_ctx(new_chat_ctx))
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)


class TechnicalQAStudentAgent(BaseStudentAgent):
    def __init__(
        self,
        ctx: JobContext,
        interview_lang: Literal["en", "ja"] = "en",
        seed_chat_ctx_filepath: str | None = None,
        ctx_max_n: int = 25,
        ctx_trim_ratio: float = 0.7,
    ) -> None:
        super().__init__(
            ctx=ctx,
            interview_lang=interview_lang,
            seed_chat_ctx_filepath=seed_chat_ctx_filepath,
            ctx_max_n=ctx_max_n,
            ctx_trim_ratio=ctx_trim_ratio,
        )

    @override
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        self.session.userdata.curr_section = "Technical_QA"

    @function_tool
    async def transfer_to_behavioral_qa(self, context: RunContext) -> Agent:  # noqa
        """Use this tool when user explicitly said that they want to end the technical interview section and you will proceed to the behavioral interview section."""  # noqa

        return BehavioralQAStudentAgent(
            ctx=self.session.userdata.ctx,
            chat_ctx=self.chat_ctx.copy(),
            interview_lang=self.interview_lang,
            ctx_max_n=self.ctx_max_n,
            ctx_trim_ratio=self.ctx_trim_ratio,
        )


class BehavioralQAStudentAgent(BaseStudentAgent):
    def __init__(
        self,
        ctx: JobContext,
        chat_ctx: ChatContext,
        interview_lang: Literal["en", "ja"] = "en",
        ctx_max_n: int = 25,
        ctx_trim_ratio: float = 0.7,
    ) -> None:
        super().__init__(
            ctx=ctx,
            interview_lang=interview_lang,
            chat_ctx=chat_ctx,
            ctx_max_n=ctx_max_n,
            ctx_trim_ratio=ctx_trim_ratio,
        )

    @override
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        if self.chat_ctx is not None:
            chat_ctx_copy = self.chat_ctx.copy()
            new_chat_ctx = chat_ctx_copy.truncate(max_items=20)
            logger.info(f"[{agent_name}] Forced context truncation")
            await self.update_chat_ctx(new_chat_ctx)

        self.session.userdata.curr_section = "Behavioral_QA"
        self.session.generate_reply()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    userdata = UserData(ctx=ctx, curr_section="Unknown")
    agent = TechnicalQAStudentAgent(
        ctx=ctx,
        interview_lang="en",
        # seed_chat_ctx_filepath="chat_ctx_dump/seed_conversation.json",
        ctx_max_n=8,
        ctx_trim_ratio=0.7,
    )
    chat_ctx_dumper = ChatCtxDumper(logger=logger, output_folder=CHAT_CTX_DUMP_FOLDER, agent=agent)

    session: AgentSession[UserData] = AgentSession(userdata=userdata)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        logger.info(f"Conversation item added from {event.item.role}")  # type: ignore
        if session._agent is not None:
            chat_ctx_dumper.agent = session._agent  # type: ignore
            chat_ctx_dumper.dump_chat_ctx()
            # session._agent.context_management()  # type: ignore
            session_context_manager(session)

        # check if truncated messages persist in the history
        logger.info(f"N session history: {len(session.history.items)}")


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
