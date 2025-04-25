import json
import logging
import os
from datetime import datetime, timedelta, timezone

from livekit.agents import Agent, ChatContext


def get_datetime_str(use_jst: bool = False) -> str:
    if use_jst:
        tz_jst = timezone(timedelta(hours=+9), "JST")
        curr_dt = datetime.now(tz=tz_jst)
        return f"{curr_dt.strftime('%Y%m%d_%H%M%S')}_{curr_dt.strftime('%f')[:3]}"
    else:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{datetime.now().strftime('%f')[:3]}"


class ChatCtxDumper:
    def __init__(self, logger: logging.Logger, output_folder: str, agent: Agent | None) -> None:
        self.agent = agent
        self.logger = logger
        self.output_folder = output_folder
        self.turn = 0
        os.makedirs(self.output_folder, exist_ok=True)

    def dump_chat_ctx(self) -> None:
        if self.agent is not None:
            output_path = f"{self.output_folder}/chat_ctx_{self.turn}.json"
            self.logger.info(f"Dump chat ctx to {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.agent.chat_ctx.copy().to_dict(), ensure_ascii=False))

            self.turn += 1


def load_chat_ctx_from_file(file_path: str) -> ChatContext:
    with open(file_path, "r", encoding="utf-8") as f:
        chat_ctx = ChatContext.from_dict(json.loads(f.readline()))
    return chat_ctx
