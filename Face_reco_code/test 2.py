import Key_for_openAi
from openai import OpenAI

client = OpenAI(
    api_key=Key_for_openAi.OPENAI_API_KEY
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
)