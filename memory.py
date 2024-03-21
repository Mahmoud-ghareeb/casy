from typing import List
from langchain.memory import ConversationBufferWindowMemory


def create_memory(k: int):
    return ConversationBufferWindowMemory(k=1, return_messages=False)


def create_history(documents: List):
    history = ""
    for i in documents:
        if i.type == "human":
            history += f"human: {i.content}\n"
        else:
            history += f"ai: {i.content}\n"

    return history


def create_context(documents: List):
    context = ""
    for i in documents:
        context += i.page_content + "\n"

    return context