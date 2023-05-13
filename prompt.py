import os
import anthropic
from enum import Enum
from dotenv import load_dotenv
load_dotenv()


anthropic_client = anthropic.Client(os.getenv('ANTHROPIC_API_KEY'))


class Model(Enum):
    claude_v1_0 = "claude-v1.0"
    claude_v1_2 = "claude-v1.2"
    claude_v1_3 = "claude-v1.3"
    claude_v1_3_100k = "claude-v1.3-100k"
    claude_v1_latest = "claude-v1"
    claude_v1_latest_100k = "claude-v1-100k"
    claude_instant_v1_0 = "claude-instant-v1.0"
    claude_instant_v1_1 = "claude-instant-v1.1"
    claude_instant_v1_1_100k = "claude-instant-v1.1-100k"
    claude_instant_v1_latest = "claude-instant-v1"
    claude_instant_v1_latest_100k = "claude-instant-v1-100k"


class Prompt:
    def __init__(
        self,
        human_message: str,
        model: Model = Model.claude_v1_latest,
        temp_0_1: float = 0.5,
        max_tokens_to_sample: int = 1024,
        assistant_prefix: str = None,
        response_prefix: str = None
    ):
        self.human_message = human_message
        self.model = model
        self.temp_0_1 = temp_0_1
        self.max_tokens_to_sample = max_tokens_to_sample
        self.assistant_prefix = assistant_prefix
        self.response_prefix = response_prefix

    @property
    def prompt(self) -> str:
        prompt = f"{anthropic.HUMAN_PROMPT} {self.human_message}{anthropic.AI_PROMPT}"
        if self.assistant_prefix:
            prompt += f" {self.assistant_prefix}"
        return prompt

    def get_response(self) -> str:
        response = anthropic_client.completion(
            prompt=self.prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model.value,
            max_tokens_to_sample=self.max_tokens_to_sample,
            temperature=self.temp_0_1,
        )

        text = response['completion']

        if self.response_prefix:
            text = f"{self.response_prefix}{text}"

        return text
