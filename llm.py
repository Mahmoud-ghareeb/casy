from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


def openai_llm(prompt, ):
    opena_api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=opena_api_key)

    messages = [
        {"role": "system", "content": prompt.template}
    ]

    response = client.chat.completions.create(
            model = "gpt-3.5-turbo-1106",
            temperature= 0,
            messages=messages,
            stream=True
        )

    res = ""

    for chunk in response:
        txt = chunk.choices[0].delta.content
        res += txt if txt is not None else ""

        yield txt if txt is not None else ""
        
    

