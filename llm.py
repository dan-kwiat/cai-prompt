import os
import anthropic
import openai
from enum import Enum
from dotenv import load_dotenv
load_dotenv()


anthropic_client = anthropic.Client(os.getenv('ANTHROPIC_API_KEY'))
openai.api_key = os.getenv("OPENAI_API_KEY")


class Model(Enum):
    claude_v1_latest = 'claude-v1'
    claude_v1_0 = "claude-v1.0"
    claude_v1_2 = "claude-v1.2"
    claude_v1_3 = "claude-v1.3"
    claude_instant_v1_latest = "claude-instant-v1"
    claude_instant_v1_0 = "claude-instant-v1.0"
    text_davinci_003 = 'text-davinci-003'
    gpt_3_5_turbo = 'gpt-3.5-turbo'
    gpt_4 = 'gpt-4'


def get_llm_response(prompt: str, model: Model = Model.claude_v1_latest, temp_0_1: float = 0.5):
    # OpenAI uses a temperature range from 0 to 2:
    temp_0_2 = temp_0_1 * 2

    if (model.value == 'text-davinci-003'):
        response = openai.Completion.create(
            model=model.value,
            prompt=f"Human: {prompt}\n\nAssistant: ",
            max_tokens=1024,
            temperature=temp_0_2
        )
        return response['choices'][0]['text'].strip()
    elif (model.value in ['gpt-3.5-turbo', 'gpt-4']):
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
        response = anthropic_client.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model.value,
            max_tokens_to_sample=1024,
            temperature=temp_0_1,
        )
        return response['completion']
