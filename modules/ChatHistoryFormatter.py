from langchain.schema import HumanMessage, AIMessage
class ChatHistoryFormatter:
    @staticmethod
    def format_chat_history(chat_history, len_history=10):
        formatted_history = []
        if len(chat_history) > 0:
            for human, ai in chat_history[-len_history:]:
                formatted_history.append(HumanMessage(content=human))
                formatted_history.append(AIMessage(content=ai))
            return formatted_history
        else:
            return chat_history