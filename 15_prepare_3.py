import os

from langchain.text_splitter import SpacyTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

pdf_file = "./sample.pdf"
if os.path.isfile(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    documents = loader.load()

    text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ja_core_news_sm")
    splitted_documents = text_splitter.split_documents(documents)

    splitted_documents = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="llama3", base_url=os.environ["OLLAMA_BASE_URL"]
    )

    database = Chroma(persist_directory="./.data", embedding_function=embeddings)

    database.add_documents(splitted_documents)

    print("Completed to create a database.")
else:
    print(
        "Please download sample.pdf from https://raw.githubusercontent.com/harukaxq/langchain-book/master/asset/sample.pdf"
    )
