import os
import openai
import anthropic
from dotenv import load_dotenv
from enum import Enum, auto
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Client(os.getenv('ANTHROPIC_API_KEY'))


class Model(Enum):
    text_davinci_003 = 'text-davinci-003'
    gpt_3_5_turbo = 'gpt-3.5-turbo'
    gpt_4 = 'gpt-4'
    claude_v1_latest = 'claude-v1'
    claude_v1_0 = "claude-v1.0"
    claude_v1_2 = "claude-v1.2"
    claude_v1_3 = "claude-v1.3"
    claude_instant_v1_latest = "claude-instant-v1"
    claude_instant_v1_0 = "claude-instant-v1.0"


def get_llm_response(prompt: str, model: Model = Model.claude_v1_latest, temp_0_1: float = 0.5):
    # OpenAI uses a temperature range from 0 to 2:
    temp_0_2 = temp_0_1 * 2

    if (model.value == 'text-davinci-003'):
        # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        response = openai.Completion.create(
            model=model.value,
            prompt=f"Human: {prompt}\n\nAssistant: ",
            max_tokens=1024,
            temperature=temp_0_2
        )
        return response['choices'][0]['text'].strip()
    elif (model.value in ['gpt-3.5-turbo', 'gpt-4']):
        # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        response = openai.ChatCompletion.create(
            model=model.value,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
            temperature=temp_0_2,
            stream=False
        )
        return response['choices'][0]['message']['content']
    else:
        # Amount of randomness injected into the response. Ranges from 0 to 1. Use temp closer to 0 for analytical / multiple choice, and temp closer to 1 for creative and generative tasks.
        response = anthropic_client.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model.value,
            max_tokens_to_sample=1024,
            temperature=temp_0_1,
        )
        return response['completion']
