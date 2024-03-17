from memory import create_memory, create_context, create_history
from embedding import load_and_embedd, encode
from llm import openai_llm
from audio import get_audio_stream
from args import Args

from langchain.prompts import PromptTemplate

class Casy:

    def __init__(self):
        self.g_vars = {}
        
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
        return openai_llm(prompt)
    
    def stream_audio(self, question):
        prompt = self.get_the_prompt(question)
        return get_audio_stream(openai_llm(prompt))
        
