import os
import sys

import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.text_splitter import SpacyTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage

datapath = "./.data"

if not os.path.exists(datapath):
    print("Please run 15_prepare_3.py to create database")
    sys.exit(1)

embeddings = OllamaEmbeddings(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください.
                        文章:
                        {document}

                        質問: {query}""",
    input_variables=["document", "query"],
)

text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ja_core_news_sm")


@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="PDFを選択してください",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    file = files[0]
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    documents = PyMuPDFLoader(file.path).load()
    splitted_documents = text_splitter.split_documents(documents)
    database = Chroma(
        embedding_function=embeddings,
    )
    database.add_documents(splitted_documents)

    cl.user_session.set("database", database)
    await cl.Message(content=f"`{file.name}`の読み込みが完了しました．質問を入力してください").send()


@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ: " + input_message.content)

    database = cl.user_session.get("database")
    documents = database.similarity_search(input_message.content)
    document_string = ""
    for document in documents:
        document_string += f"""
        ------------------------
        {document.page_content}
        """

    result = chat.invoke(
        [
            HumanMessage(
                content=prompt.format(
                    document=document_string, query=input_message.content
                )
            )
        ]
    )
    await cl.Message(content=str(result.content)).send()
