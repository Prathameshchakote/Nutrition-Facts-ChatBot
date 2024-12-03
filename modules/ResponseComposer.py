from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
class AnswerGenerator:
    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.prompt_template = """You are an assistant for question-answering tasks regarding
                                    plant-based nutrition to prevent and reverse disease and nutrition science in general. 
                                    The following are snippets of transcripts from Dr Michael Greger's nutrition facts
                                    videos that you are going to use to to answer the question. If you don't know the answer, just say that you don't know.
                                    Ask follow-up questions if appropriate. 

                                    Context: {context}

                                    Question: {contextualized_question}

                                    Answer: 
                                    """
        self.prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        self.qa_chain = self.prompt_template | self.chat_model | StrOutputParser()

    def generate_answer(self, context, question):
        self.context = context
        self.question = question
        self.answer = self.qa_chain.invoke({'context': self.context, 'contextualized_question': self.question})
        return self.answer