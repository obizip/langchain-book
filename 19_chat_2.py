import os
import sys

import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
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

database = Chroma(persist_directory=datapath, embedding_function=embeddings)


@cl.on_chat_start
async def on_caht_start():
    await cl.Message(content="準備ができました!メッセージを入力してください!").send()


@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ: " + input_message.content)

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
