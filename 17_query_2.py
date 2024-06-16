import os

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage

datapath = "./.data"

if os.path.exists(datapath):
    embeddings = OllamaEmbeddings(
        model="llama3", base_url=os.environ["OLLAMA_BASE_URL"]
    )

    database = Chroma(persist_directory=datapath, embedding_function=embeddings)

    query = "飛行車の最高速度は?"
    documents = database.similarity_search(query)
    document_string = ""
    for document in documents:
        document_string += f"""
        ------------------------
        {document.page_content}
        """
    prompt = PromptTemplate(
        template="""文章を元に質問に答えてください.
                            文章:
                            {document}

                            質問: {query}""",
        input_variables=["document", "query"],
    )

    chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

    result = chat.invoke(
        [HumanMessage(content=prompt.format(document=document_string, query=query))]
    )
    print(result.content)

else:
    print("Please run 15_prepare_3.py to create database")
