from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from ContextRetriever import ContextRetriever
from ChatHistoryFormatter import ChatHistoryFormatter
from QuestionContextualizer import QuestionContextualizer
from AnswerGenerator import AnswerGenerator
from ResultFormatter import ResultFormatter
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
import os

class NutritionBot:
    def __init__(self, temp=0.5, 
                       chat_model_name='gpt-4-turbo-2024-04-09', 
                       embeddings_model_name='text-embedding-3-large',
                       num_results=10,
                       persist_directory='../chroma_db'):
        # Load environment and API keys
        load_dotenv()
        api_key = os.environ['OPENAI_API_KEY']

        self.embeddings_model_name = embeddings_model_name
        self.temp = temp
        self.chat_model_name = chat_model_name
        self.num_results = num_results
        self.persist_directory = persist_directory

        # Embeddings Model
        self.embeddings_model = OpenAIEmbeddings(model=self.embeddings_model_name, max_retries=100, chunk_size=16, show_progress_bar=False)

        # Initialize Chat Model
        self.chat_model = ChatOpenAI(temperature=self.temp, model=self.chat_model_name)

        # Set up vector store and other components
        self.context_retriever = ContextRetriever(embeddings_model=self.embeddings_model, persist_directory=self.persist_directory, num_results=self.num_results)
        self.chat_history_formatter = ChatHistoryFormatter()
        self.question_contextualizer = QuestionContextualizer(self.chat_model)
        self.answer_generator = AnswerGenerator(self.chat_model)
        self.result_formatter = ResultFormatter()

    def process_chat(self, new_question, chat_history):
        formatted_chat_history = self.chat_history_formatter.format_chat_history(chat_history)
        contextualized_question = self.question_contextualizer.contextualize_question(formatted_chat_history, new_question)
        context = self.context_retriever.get_context(contextualized_question)
        answer = self.answer_generator.generate_answer(context=context, question=contextualized_question)
        final_result = self.result_formatter.format_result(answer, context)
        return final_result