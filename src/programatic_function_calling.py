# Example code for programmatically creating a function calling agent with LiveKit
# 1. Normal programatic function creation
# 2. Make a certain function available only after another function has been called
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero
from livekit.plugins.openai.realtime.realtime_model import InputAudioTranscription
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv()


@dataclass
class UserData:
    """Stores data and agents to be shared across roleplay sections."""

    add_answer_company_info_question: bool = False
    function_already_added: bool = False


def get_instruction(interview_language: Literal["Japanese", "English"] = "Japanese") -> str:
    system_prompt = "".join(
        [
            "You are a 22-year-old, final-year Japanese university student"
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


def company_info():
    async def given_company_info(context: RunContext, info: str):  # noqa
        """Called when the interviewer explains about the company.

        :param str info: company information or explanation. This is required.
        """
        logger.info(f"Company Info: {info}")
        context.userdata.add_answer_company_info_question = True
        return {"company info": info}

    return given_company_info


def answer_company_info_question():
    async def answer_company_question(answer: str) -> None:  # noqa
        """Called when the interviewer answers your question regarding the company."""
        logger.info("Answering company question")
        return {"interviewer_answer": answer}

    return answer_company_question


class SampleAgent(Agent):
    def __init__(self, ctx: JobContext, interview_language: str = "English") -> None:
        """Constructor."""
        super().__init__(
            instructions=get_instruction(interview_language),  # type: ignore
            allow_interruptions=False,
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            stt=openai.STT(language="en", model="gpt-4o-mini-transcribe"),
            llm=openai.realtime.RealtimeModel(
                model="gpt-4o-realtime-preview-2024-12-17",
                voice="sage",
                turn_detection=None,
                input_audio_transcription=InputAudioTranscription(
                    language="en", model="gpt-4o-mini-transcribe"
                ),
            ),
        )

    async def on_enter(self) -> None:
        logger.info(f"Entering {self.__class__.__name__}")
        self.session.generate_reply(instructions="Greet the interviewer Mr. Tamura.")

    @function_tool
    async def how_are_you(self) -> dict[str, Any]:  # noqa
        """Called when the user asks about your condition. Example questions: "How are you?", "I'm great how about you?", "How is it going?"."""  # noqa
        return {"message": "I am not very well due to the recent rainy days?"}

    def print_tools(self) -> None:
        """Prints the tools available to the agent."""
        for tool in self.tools:
            logger.info(f"Available Tool: Dict: {tool.__dict__}")


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # store reference to asyncio tasks
    background_tasks = set()

    # agent
    agent = SampleAgent(ctx=ctx)
    agent.print_tools()

    # add new tool
    new_tool = function_tool(
        company_info(),
        name="given_company_info",
        description="Called when the interviewer explains about the company.",
    )
    await agent.update_tools(agent.tools + [new_tool])  # noqa
    agent.print_tools()

    # session
    session: AgentSession[UserData] = AgentSession(userdata=UserData())
    await session.start(agent=agent, room=ctx.room)

    # audio
    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
        # play keyboard typing sound when the agent is thinking
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.8),
        ],
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        logger.info(f"Conversation item added from {event.item.role}")  # type: ignore
        if (
            session.userdata.add_answer_company_info_question
            and not session.userdata.function_already_added
        ):
            logger.info("Add 'answer_company_question' tool on the fly")

            new_tool = function_tool(
                answer_company_info_question(),
                name="answer_company_info_question",
                description="Called when the interviewer answers your "
                + "question regarding the company.",
            )
            task = asyncio.create_task(
                session._agent.update_tools(session._agent.tools + [new_tool])  # noqa
            )
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

            # flag so the system not add the function again
            session.userdata.function_already_added = True


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
