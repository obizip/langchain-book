import os
import sys

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

datapath = "./.data"
if not os.path.exists(datapath):
    print("Please run 15_prepare_3.py to create database")
    sys.exit(1)


chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

embeddings = OllamaEmbeddings(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

database = Chroma(persist_directory=datapath, embedding_function=embeddings)

retriever = database.as_retriever()

qa = RetrievalQA.from_llm(llm=chat, retriever=retriever, return_source_documents=True)

result = qa.invoke("飛行車の最高速度を教えて")

print(result["result"])
print(result["source_documents"])
