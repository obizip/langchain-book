import os

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

datapath = "./.data"

if os.path.exists(datapath):
    embeddings = OllamaEmbeddings(
        model="llama3", base_url=os.environ["OLLAMA_BASE_URL"]
    )

    database = Chroma(persist_directory=datapath, embedding_function=embeddings)

    documents = database.similarity_search("飛行車の最高速度は?")
    print(f"ドキュメントの数: {len(documents)}")

    for document in documents:
        print(f"ドキュメントの内容: {document.page_content}")
else:
    print("Please run 15_prepare_3.py to create database")
