import os

from langchain_community.document_loaders import PyMuPDFLoader

pdf_file = "./sample.pdf"
if os.path.isfile(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    documents = loader.load()

    print(f"the number of documents: {len(documents)}")
    print(f"contents of first document: {documents[0].page_content}")
    print(f"meta data of first documnts: {documents[0].metadata}")
else:
    print(
        "Please download sample.pdf from https://raw.githubusercontent.com/harukaxq/langchain-book/master/asset/sample.pdf"
    )
