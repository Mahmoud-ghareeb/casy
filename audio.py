from typing import Iterator
import subprocess
import elevenlabs
import os

from dotenv import load_dotenv

load_dotenv()
elevenlabs.set_api_key(os.getenv('ELEVEN_LAPS_API_KEY'))

def get_audio_stream(llm):
    audio_stream = elevenlabs.generate(text=llm, voice="tsample", model="eleven_multilingual_v2", stream=True)
    for chunck in audio_stream:
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