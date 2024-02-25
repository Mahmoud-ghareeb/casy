from embedding import load_and_embedd
from args import Args
from schema import Message

from fastapi import FastAPI, File, UploadFile
import shutil


app = FastAPI()
args = Args()

@app.post("/")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"books/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    app.db, app.embeddings = load_and_embedd(file_location, args.model_id['paraphrase-MiniLM'])

    return {"filename": file.filename, "location": file_location}

@app.post("/chat")
async def message(message: Message):
    
    question_embed = app.embeddings.embed_query(message.text)
    results = app.dp.query(
        query_embeddings=question_embed.tolist(),
        n_results=10,  
    )

    return {"response": message.text, "add": results}