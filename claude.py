from typing import NamedTuple, Optional
import os
import anthropic
from enum import Enum
from dotenv import load_dotenv
load_dotenv()


anthropic_client = anthropic.Client(os.getenv('ANTHROPIC_API_KEY'))


class Model(Enum):
    claude_v1_latest = 'claude-v1'
    claude_v1_0 = "claude-v1.0"
    claude_v1_2 = "claude-v1.2"
    claude_v1_3 = "claude-v1.3"
    claude_instant_v1_latest = "claude-instant-v1"
    claude_instant_v1_0 = "claude-instant-v1.0"


class PromptType(NamedTuple):
    human_message: str
    model: Optional[Model] = Model.claude_v1_latest
    temp_0_1: Optional[float] = 0.5
    max_tokens_to_sample: Optional[int] = 1024
    assistant_prefix: Optional[str] = None
    response_prefix: Optional[str] = None


def get_claude_response(
    prompt_args: PromptType
) -> str:
    wrapped_prompt = f"{anthropic.HUMAN_PROMPT} {prompt_args.human_message}{anthropic.AI_PROMPT}"

    if (prompt_args.assistant_prefix is not None):
        wrapped_prompt += f" {prompt_args.assistant_prefix}"

    response = anthropic_client.completion(
        prompt=wrapped_prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=prompt_args.model.value,
        max_tokens_to_sample=prompt_args.max_tokens_to_sample,
        temperature=prompt_args.temp_0_1,
    )

    text = response['completion']

    if (prompt_args.response_prefix is not None):
        # This is often useful to clean up after we've used `assistant_prefix` e.g. to start a numbered list of items.
        text = f"{prompt_args.response_prefix}{text}"

    return text
