from embedding import load_and_embedd, encode
from memory import create_memory, create_context, create_history
from llm import openai_llm
from langchain.prompts import PromptTemplate
from args import Args
from schema import Message

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import shutil
import yaml

args = Args()

g_vars = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = encode(args.model_id['paraphrase-MiniLM'], device=args.device)
    memory = create_memory(args.k)
    config = yaml.load(open("configs/config.default.yaml", "r"), Loader=yaml.FullLoader)

    g_vars['embedding'] = embeddings
    g_vars['dp'] = ''
    g_vars['memory'] = memory
    g_vars['config'] = config
    yield
    # g_vars.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/")
async def create_upload_file(file: UploadFile = File(...)):
    """ Create embedding of the uploaded book

    :param file: the uploaded file (.docs, .pdf)
    """
    file_location = f"books/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    g_vars['dp'] = load_and_embedd(file_location, g_vars['embedding'])

    return {"filename": file.filename, "location": file_location}


@app.post("/chat")
async def message(message: Message):
    
    results = g_vars['dp'].max_marginal_relevance_search(message.text)
    template = g_vars['config']['prompts']['initial_propmt']
    context = create_context(results)
    history = create_history(g_vars['memory'].chat_memory.messages)
    question = message.text

    new_template = template.format(
        context=context,
        history=history,
        question=question
    )
    prompt = PromptTemplate(
        input_variables=["context", "history", "question"], 
        template=new_template
    )

    return StreamingResponse(openai_llm(prompt), media_type="text/plain")