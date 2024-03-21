import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import Iterator
import subprocess
import elevenlabs
import os
import json

from dotenv import load_dotenv

load_dotenv()
elevenlabs.set_api_key(os.getenv('ELEVEN_LAPS_API_KEY'))
executor = ThreadPoolExecutor()

def get_audio_stream(llm):
    audio_stream = elevenlabs.generate( text=llm,
                                        voice='tsample', 
                                        model="eleven_multilingual_v2", 
                                        stream=True)
    # stream(audio_stream)
   
    for chunck in audio_stream:
        if chunck:
            yield chunck

async def get_text_audio_stream(llm):
    loop = asyncio.get_running_loop()
    audio_stream = await loop.run_in_executor(executor, lambda: list(elevenlabs.generate(text=llm, voice="tsample", model="eleven_multilingual_v2", stream=True)))
    async for chunck in audio_stream:
        if chunck:
            yield chunck

def stream(audio_stream: Iterator[bytes]) -> bytes:
    mpv_command = ["C:\\Program Files\\mpv\\mpv.exe", "--no-cache", "--no-terminal", "--", "fd://0"]
    mpv_process = subprocess.Popen(
        mpv_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    audio = b""

    for chunk in audio_stream:
        if chunk is not None:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()
            audio += chunk

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()

    return audio