import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

from .vector_db.vector_db import VectorDB


class ChatBotLangChain:
    PROJECT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "../datas/prompt")

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, max_tokens=1024, model="gpt-3.5-turbo-16k")
        self.vectorDB = VectorDB()
        self.api_name = ""

    def read_prompt_template(self, template: str) -> str:
        file_path = os.path.join(self.PROJECT_PROMPT_PATH, template + ".txt")
        with open(file_path, "r") as f:
            prompt_template = f.read()

        return prompt_template

    def create_chain(self, llm, template, output_key):
        return LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(
                template=self.read_prompt_template(template),
            ),
            output_key=output_key,
            verbose=True,
        )

    def separate_intent(self, text):
        self.api_name = self.create_chain(self.llm, "parse_intent", "intent").run(
            dict(text=text, intent_list=self.read_prompt_template("intent_list")))

    def generate_output(self, text, context):
        self.separate_intent(text)
        if self.vectorDB.CHROMA_COLLECTION_NAME.get(self.api_name) is None:
            context["output"] = self.create_chain(self.llm, "prompt_without_docs", "output").run(context)
            context["docs"] = ""
            return context

        docs = self.vectorDB.query_db(text, self.api_name)
        context["docs"] = docs
        context["output"] = self.create_chain(self.llm, "prompt", "output").run(context)
        return context
