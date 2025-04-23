import logging
from typing import Literal

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import openai, silero
from openai.types.beta.realtime.session import TurnDetection

logger = logging.getLogger("realtime-turn-detector")
logger.setLevel(logging.INFO)

load_dotenv()


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


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session: AgentSession = AgentSession(
        llm=openai.realtime.RealtimeModel(
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=True,
                interrupt_response=True,
            )
        )
    )
    await session.start(agent=Agent(instructions=get_instruction("Japanese")), room=ctx.room)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
