import random

from .chatbot_langchain import ChatBotLangChain
from .chatbot_history import ChatBotHistory


class ChatBotService:
    def __init__(self):
        self.langChain = ChatBotLangChain()
        self.history = ChatBotHistory()
        self.conversation_id = random.randbytes(3).hex()

    def generate_output(self, text):
        history_file = self.history.load_conversation_history(self.conversation_id)
        context = dict(chat_history=self.history.get_chat_history(self.conversation_id), text=text)
        context = self.langChain.generate_output(text, context)
        self.history.log_all(history_file, context)
        return context["output"]

# if __name__ == "__main__":
#     c = ChatBotService()
#     d1 = c.generate_output("카카오 싱크에 어떤 기능 목록이 있나요?")
#     d2 = c.generate_output("오늘 날씨 어때?")
#     print(d1)
#     print(d2)
