"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.

import pynecone as pc
from pynecone.base import Base

from .chatbot_service import ChatBotService

# openai.api_key = "<YOUR_OPENAI_API_KEY>"
api_key = ""


class Message(Base):
    text: str
    role: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    def send(self):
        self.messages.append(Message(text=self.text, role="user"))
        self.messages.append(Message(text=ChatBotService(api_key).generate_output(self.text), role="assistant"))
        State.set_text("")


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Chat-Bot 🗺", font_size="2rem")
    )


def text_box(text, color):
    return pc.text(
        text,
        background_color=color,
        padding="1rem",
        border_radius="8px",
    )


def message(msg):
    return pc.box(
        text_box(msg.role + ":\n" + msg.text, "#7ab0ff"),
        spacing="0.3rem",
        align_items="left",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        pc.hstack(
            pc.input(
                placeholder="Ask any question...",
                on_blur=State.set_text,
                border_color="#eaeaef"
            ),
            pc.button("Send", on_click=State.send, margin_top="1rem"),
            padding="2rem",
            max_width="600px"
        )
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="ChatBot")
app.compile()

