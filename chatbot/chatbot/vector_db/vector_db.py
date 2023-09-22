import os

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = ""


class VectorDB:
    PROJECT_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../datas/project_data")
    CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "persist")
    CHROMA_COLLECTION_NAME = {"카카오싱크": "kakao-sync",
                              "카카오소셜": "kakao-social",
                              "카카오톡채널": "kakao-channel"}

    def __init__(self):
        if not os.path.isdir(self.CHROMA_PERSIST_DIR):
            os.mkdir(self.CHROMA_PERSIST_DIR)
            self.upload_project_datas()

    def upload_project_datas(self):
        for k in self.CHROMA_COLLECTION_NAME:
            self.upload_project_data(k)

    def upload_project_data(self, api_name):
        loader = TextLoader(os.path.join(self.PROJECT_DATA_PATH, "project_data_" + api_name + ".txt"))
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0, separator="#")
        docs = text_splitter.split_documents(documents)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200, separator="\n")
        docs = text_splitter.split_documents(docs)

        Chroma.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
            collection_name=self.CHROMA_COLLECTION_NAME.get(api_name),
            persist_directory=self.CHROMA_PERSIST_DIR,
        )

    def db(self, api_name):
        return Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=self.CHROMA_PERSIST_DIR,
            collection_name=self.CHROMA_COLLECTION_NAME.get(api_name),
        )

    def query_db(self, query, api_name):
        docs = self.db(api_name).as_retriever().get_relevant_documents(query)
        return [doc.page_content for doc in docs]


if __name__ == "__main__":
    v = VectorDB()
    d1 = v.query_db("카카오 싱크 기능 목록", "카카오싱크")
    d2 = v.query_db("카카오 싱크 기능 목록", "카카오소셜")
    print(d1)
    print(d2)
