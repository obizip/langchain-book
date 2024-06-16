import os

from langchain.text_splitter import SpacyTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# python3 -m spacy download ja_core_news_sm
pdf_file = "./sample.pdf"
if os.path.isfile(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    documents = loader.load()

    text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ja_core_news_sm")
    splitted_documents = text_splitter.split_documents(documents)

    print(f"Before split documents: {len(documents)}")
    print(f" After split documents: {len(splitted_documents)}")
else:
    print(
        "Please download sample.pdf from https://raw.githubusercontent.com/harukaxq/langchain-book/master/asset/sample.pdf"
    )
