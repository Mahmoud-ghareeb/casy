import asyncio
from concurrent.futures import ThreadPoolExecutor

from memory import create_memory, create_context, create_history
from embedding import load_and_embedd, encode
from audio import get_audio_stream, get_text_audio_stream
from args import Args
from llm import openai_llm

import elevenlabs

from langchain.prompts import PromptTemplate

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class Casy:

    def __init__(self):
        self.g_vars = {}
        opena_api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=opena_api_key)
        self.aclient = AsyncOpenAI(api_key=opena_api_key)
        elevenlabs.set_api_key(os.getenv('ELEVEN_LAPS_API_KEY'))
        self.executor = ThreadPoolExecutor()
        
    def get_the_prompt(self, question):
        results = self.g_vars['dp'].max_marginal_relevance_search(question, k=1)
        template = self.g_vars['config']['prompts']['test_propmt']
        context = create_context(results)
        history = create_history(self.g_vars['memory'].chat_memory.messages)

        new_template = template.format(
            context=context,
            history=history,
            question=question
        )
        prompt = PromptTemplate(
            input_variables=["context", "history", "question"], 
            template=new_template
        )

        return prompt
    
    def stream_text(self, question):
        prompt = self.get_the_prompt(question)
        print(prompt)
        return self.openai_llm(prompt, question)
    
    async def stream_text_ws(self, question, websocket):
        prompt = self.get_the_prompt(question)
        print(prompt)
        messages = [
            {"role": "system", "content": prompt.template}
        ]

        response = await self.aclient.chat.completions.create(
                model = "gpt-3.5-turbo-1106",
                temperature= 0,
                messages=messages,
                stream=True,
            )

        res = ""

        async for chunk in response:
            txt = chunk.choices[0].delta.content
            txt = txt if txt is not None else ""
            res += txt 
            
            await websocket.send_text(txt)

            yield txt

        self.g_vars['memory'].chat_memory.messages = []
        self.g_vars['memory'].save_context({"input": question}, {"output": res})
    
    def stream_audio(self, question):
        prompt = self.get_the_prompt(question)
        return get_audio_stream(self.openai_llm(prompt, question))
    
    async def stream_text_audio_ws(self, question, websocket):
        prompt = self.get_the_prompt(question)
        audio_stream = elevenlabs.generate(text=self.openai_llm(prompt, question), voice="tsample", model="eleven_multilingual_v2", stream=True)
        for chunk in audio_stream:
            if chunk:
                yield chunk
        # loop = asyncio.get_running_loop()
        # llm = self.openai_llm(question, websocket)
        # try:
        #     audio_stream = await loop.run_in_executor(self.executor, lambda: list(elevenlabs.generate(text=llm,
        #                                                                                      voice="tsample",
        #                                                                                      model="eleven_multilingual_v2", 
        #                                                                                      stream=True)))
        
        #     async for chunck in audio_stream:
        #         if chunck:
        #             yield chunck
        # except:
        #     print("none")
    
    def openai_llm(self, prompt, question):

        messages = [
            {"role": "system", "content": prompt.template}
        ]

        response = self.client.chat.completions.create(
                model = "gpt-3.5-turbo-1106",
                temperature= 0,
                messages=messages,
                stream=True,
            )

        res = ""

        for chunk in response:
            txt = chunk.choices[0].delta.content
            res += txt if txt is not None else ""
            
            yield txt if txt is not None else ""

        self.g_vars['memory'].chat_memory.messages = []
        self.g_vars['memory'].save_context({"input": question}, {"output": res})

    async def aopenai_llm(self, prompt, question):

        messages = [
            {"role": "system", "content": prompt.template}
        ]

        response = await self.aclient.chat.completions.create(
                model = "gpt-3.5-turbo-1106",
                temperature= 0,
                messages=messages,
                stream=True,
            )

        res = ""

        async for chunk in response:
            txt = chunk.choices[0].delta.content
            txt = txt if txt is not None else ""
            res += txt 
            
            # await websocket.send_text(txt)

            yield txt

        self.g_vars['memory'].chat_memory.messages = []
        self.g_vars['memory'].save_context({"input": question}, {"output": res})


    # async def stream_text_audio(self, question):
    #     prompt = self.get_the_prompt(question)
    #     text_chunks = self.openai_llm(prompt, question)

    #     async for text_chunk in text_chunks:
    #         if text_chunk:
    #             yield ['text', text_chunk]
                
    #             audio_stream = self.get_audio_stream(text_chunk)
    #             async for audio_chunk in audio_stream:
    #                 if audio_chunk:
    #                     yield ['audio', audio_chunk]

    def get_audio_stream(self, llm):
        audio_stream = elevenlabs.generate(text=llm, voice="tsample", model="eleven_multilingual_v2", stream=True)
        for chunk in audio_stream:
            if chunk:
                yield chunk

            
