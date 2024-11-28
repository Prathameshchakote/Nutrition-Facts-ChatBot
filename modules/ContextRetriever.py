from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Importing the existing classes from the original implementation
class ContextRetriever:
    def __init__(self, embeddings_model, persist_directory, num_results):
        self.persist_directory = persist_directory
        self.num_results = num_results
        self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': self.num_results})
    
    def get_context(self, question):
        self.context = self.retriever.invoke(question)
        return self.context