import os

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = ""


class VectorDB:
    PROJECT_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../datas/project_data")
    CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "persist")
    CHROMA_COLLECTION_NAME = "dosu-bot"

    def __init__(self):
        if not os.path.isdir(self.CHROMA_PERSIST_DIR):
            os.mkdir(self.CHROMA_PERSIST_DIR)
            self.upload_project_data()

    def upload_project_data(self):
        loader = DirectoryLoader(self.PROJECT_DATA_PATH, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0, separator="#")
        docs = text_splitter.split_documents(documents)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200, separator="\n")
        docs = text_splitter.split_documents(docs)
        Chroma.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
            collection_name=self.CHROMA_COLLECTION_NAME,
            persist_directory=self.CHROMA_PERSIST_DIR,
        )

    def db(self):
        return Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=self.CHROMA_PERSIST_DIR,
            collection_name=self.CHROMA_COLLECTION_NAME,
        )


if __name__ == "__main__":
    v = VectorDB()
    d = v.db().as_retriever().get_relevant_documents("카카오 싱크 기능 목록")
    print(d)
