import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChromaDBLoader:
    def __init__(self, api_key, embedding_model='text-embedding-3-large'):
        """
        Initialize ChromaDB loader with OpenAI embeddings
        
        :param api_key: OpenAI API key
        :param embedding_model: Embedding model to use
        """
        self.embeddings_model = OpenAIEmbeddings(
            api_key=api_key, 
            model=embedding_model, 
            max_retries=100, 
            chunk_size=16
        )
        self.vectorstore = None
        self.docs = None

    def load_csv_to_vectorstore(self, csv_path, persist_directory="../chroma_db", text_column='transcript'):
        """
        Load CSV file to ChromaDB vector store
        
        :param csv_path: Path to CSV file
        :param persist_directory: Directory to persist ChromaDB
        :param text_column: Column to use for text embedding
        """
        # Read CSV and select relevant columns
        df = pd.read_csv(csv_path)
        cols_keep = ['videoId', 'title', 'description', 'publishedAt', text_column]
        df = df[cols_keep]

        # Load documents from DataFrame
        loader = DataFrameLoader(df, page_content_column=text_column)
        self.docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
        splits = text_splitter.split_documents(self.docs)

        # Create or add to ChromaDB
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings_model, 
            persist_directory=persist_directory
        )

    def get_document_sources(self):
        """
        Retrieve unique document sources in the vectorstore
        
        :return: Set of unique document sources
        """
        if not self.docs:
            return set()
        
        # Extract sources from the original documents' metadata
        return set(doc.metadata.get('title', 'Unknown') for doc in self.docs)

    def get_retriever(self, search_kwargs=None):
        """
        Get a retriever from the vectorstore
        
        :param search_kwargs: Optional search parameters
        :return: Retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call load_csv_to_vectorstore first.")
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs or {})

def main():
    # Load environment variables
    load_dotenv()
    
    # Replace with your actual OpenAI API key from environment variable
    OPENAI_APIKEY = os.environ['OPENAI_APIKEY']
    
    if not OPENAI_APIKEY:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
    
    # Initialize loader
    loader = ChromaDBLoader(api_key=OPENAI_APIKEY)
    
    # Load CSV to ChromaDB
    csv_path = '../data/transcripts_all_2024-04-18_cleaned.csv'
    loader.load_csv_to_vectorstore(csv_path)
    
    # Print document sources
    sources = loader.get_document_sources()
    print(f"Loaded {len(sources)} unique document sources")
    print("Sources:", sources)

if __name__ == "__main__":
    main()