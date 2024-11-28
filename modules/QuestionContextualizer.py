from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser

class QuestionContextualizer:
    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.contextualize_q_system_prompt = """Given a chat history and the latest user question \
                                                which might reference context in the chat history, formulate a standalone question \
                                                which can be understood without the chat history. Do NOT answer the question, \
                                                just reformulate it if needed and otherwise return it as is."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}")
        ])
        self.contextualize_chain = self.prompt_template | chat_model | StrOutputParser()

    def contextualize_question(self, formatted_chat_history, question):
        self.question = question
        self.formatted_chat_history = formatted_chat_history
        self.contextualized_question = self.contextualize_chain.invoke({'chat_history': formatted_chat_history, 'question': question})
        return self.contextualized_question