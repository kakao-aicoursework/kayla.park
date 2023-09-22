import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

os.environ["OPENAI_API_KEY"] = "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"


class ChatBotService:
    PROJECT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "../datas/prompt")

    def __init__(self):
        self.prompt_path = os.path.join(os.path.dirname(__file__), "../datas/")

    def read_prompt_template(self, template: str) -> str:
        file_path = self.prompt_path + template + ".txt"
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

    def generate_output(self, text):
        docs = self.read_prompt_template("project_data_카카오싱크")
        writer_llm = ChatOpenAI(temperature=0.1, max_tokens=2048, model="gpt-3.5-turbo-16k")

        chain_1 = self.create_chain(writer_llm, "prompt1", "function")
        chain_2 = self.create_chain(writer_llm, "prompt2", "login_example")
        chain_3 = self.create_chain(writer_llm, "prompt3", "deployment")
        chain_4 = self.create_chain(writer_llm, "prompt4", "settings")
        chain_5 = self.create_chain(writer_llm, "prompt5", "output")

        preprocess_chain = SequentialChain(
            chains=[chain_1, chain_2, chain_3, chain_4, chain_5],
            input_variables=["docs", "text"],
            output_variables=["function", "login_example", "deployment", "settings", "output"],
            verbose=True,
        )

        context = dict(
            docs=docs,
            text=text
        )
        context = preprocess_chain(context)
        output = context["output"]
        return output
