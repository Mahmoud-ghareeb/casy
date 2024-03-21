from casy import Casy
from casy import load_and_embedd
from casy import encode
from casy import create_memory
from casy import Args

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager
from schema import Message
import shutil
import yaml
import hashlib
from typing import List
import base64

casy = Casy()
args = Args()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_text(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    def encode_audio(self, audio_bytes):
        return base64.b64encode(audio_bytes).decode('utf-8')

    async def send_text_and_audio(self, message, websocket: WebSocket):
        # response = {"text": "dsadsa", "audio": "dadsad"}
        # await websocket.send_json(response)
        for data in casy.stream_text_audio(message, websocket):
            await websocket.send_text(data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = encode(args.model_id['multilingual-e5-base'], device=args.device, use_open_ai=False)
    memory = create_memory(args.k)
    config = yaml.load(open('configs/config.default.yaml', 'r'), Loader=yaml.FullLoader)

    casy.g_vars['embedding'] = embeddings
    casy.g_vars['dp'] = ''
    casy.g_vars['memory'] = memory
    casy.g_vars['config'] = config
    yield
    casy.g_vars.clear()


app = FastAPI(lifespan=lifespan)
connection_manager = ConnectionManager()

@app.post("/")
async def create_upload_file(file: UploadFile = File(...)):
    """ Create embedding of the uploaded book

    :param file: the uploaded file (.docs, .pdf)
    """
    file_location = f"books/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    name = file.filename

    hash = hashlib.sha1(name.encode("UTF-8")).hexdigest()

    casy.g_vars['dp'] = load_and_embedd(file_location, casy.g_vars['embedding'], hash, is_json=True)

    return {"filename": file.filename, "location": file_location}


@app.post("/text")
async def message(message: Message):

    return StreamingResponse(casy.stream_text(message.text), media_type="text/plain")


@app.post("/audio")
async def message(message: Message):
    
    return StreamingResponse(casy.stream_audio(message.text), media_type="text/mpeg")

@app.websocket("/socket_audio")
async def message(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_content = data.split(":", 1)[1][:-1]
            print(message_content)

            async for audio_data in casy.stream_text_audio_ws(message_content, websocket):
                websocket.send_bytes(audio_data)

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

@app.websocket("/ws/audio")
async def message(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_content = data.split(":", 1)[1][:-1]
            print(message_content)

            casy.stream_text_audio(message_content, websocket)

            # async for audio_data in casy.stream_text_audio(message_content, websocket):
            #     await websocket.send_bytes(audio_data)

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

# @app.websocket("/ws/stream")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     # Example of streaming text and audio
#     for data in casy.stream_text_audio():
#         await websocket.send_json({"type": data[0], "content": data[1]})
        