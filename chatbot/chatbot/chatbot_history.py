import os
from pathlib import Path

from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.schema import BaseMessage


class ChatBotHistory:
    CHAT_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "../datas/chat_histories")

    def load_conversation_history(self, conversation_id: str):
        file_path = os.path.join(self.CHAT_HISTORY_PATH, f"{conversation_id}.json")
        return FileChatMessageHistory(file_path)

    def log_user_message(self, history: FileChatMessageHistory, user_message: str):
        history.add_user_message(user_message)

    def log_bot_message(self, history: FileChatMessageHistory, bot_message: str):
        history.add_ai_message(bot_message)

    def get_chat_history(self, conversation_id: str):
        history = self.load_conversation_history(conversation_id)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="user_message",
            chat_memory=history,
        )

        return memory.buffer

    def log_all(self, history: FileChatMessageHistory, context):
        self.log_user_message(history, context["text"])
        self.log_bot_message(history, context["output"] + "<reference>" + str(context["docs"]) + "</reference>")
