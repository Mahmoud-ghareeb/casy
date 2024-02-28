from casy import Casy
from casy import load_and_embedd
from casy import encode
from casy import create_memory
from casy import Args

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from schema import Message
import shutil
import yaml

casy = Casy()
args = Args()



@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = encode(args.model_id['paraphrase-MiniLM'], device=args.device)
    memory = create_memory(args.k)
    config = yaml.load(open("configs/config.default.yaml", "r"), Loader=yaml.FullLoader)

    casy.g_vars['embedding'] = embeddings
    casy.g_vars['dp'] = load_and_embedd('books/GITSample.docx', casy.g_vars['embedding'])
    casy.g_vars['memory'] = memory
    casy.g_vars['config'] = config
    yield
    casy.g_vars.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/")
async def create_upload_file(file: UploadFile = File(...)):
    """ Create embedding of the uploaded book

    :param file: the uploaded file (.docs, .pdf)
    """
    file_location = f"books/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    casy.g_vars['dp'] = load_and_embedd(file_location, casy.g_vars['embedding'])

    return {"filename": file.filename, "location": file_location}


@app.post("/text")
async def message(message: Message):

    return StreamingResponse(casy.stream_text(message.text), media_type="text/plain")


@app.post("/audio")
async def message(message: Message):
    
    return StreamingResponse(casy.stream_audio(message.text), media_type="text/mpeg")