from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
import chromadb
from functools import cache
import shutil
import os


def load_pdf(file_path):
    if not file_path.endswith('.pdf'):
        return False
    
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def read_docx(file_path):
    loader = UnstructuredWordDocumentLoader(file_path, mode="single", strategy="fast") 
    return loader.load()


def splitter(txt):
    chunk_size = 1000
    chunk_overlap = 200

    def length_function(text: str) -> int:
        return len(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function
    )

    return splitter.split_documents(txt)

@cache
def encode(model_id="", device="", use_open_ai = False):
    if use_open_ai:
        opena_api_key = os.getenv('OPENAI_API_KEY')
        return OpenAIEmbeddings(openai_api_key=opena_api_key, model="text-embedding-3-small")
    else:
        return HuggingFaceEmbeddings(model_name=model_id, model_kwargs = {"device": device})


def load_and_embedd(file_path, embeddings):
     
    if file_path.endswith('.pdf'):
        documents = load_pdf(file_path)
    else:
        documents = read_docx(file_path)
     
    splitted_txt = splitter(documents)
    persist_dir = "E:\\ro2ya_dev\\casy\\store"

    shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.Client()
    client.get_or_create_collection("casy")
    vectordb = Chroma.from_documents(
        documents=splitted_txt,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()

    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    return vector_db