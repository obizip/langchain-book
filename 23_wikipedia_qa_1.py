import os

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import WikipediaRetriever

chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

retriever = WikipediaRetriever(top_k_results=2, doc_content_chars_max=500)

chain = RetrievalQA.from_llm(
    llm=chat, retriever=retriever, return_source_documents=True
)

result = chain.invoke("What is bourbon whiskey")

source_documents = result["source_documents"]

print(f"search results: {len(source_documents)}")

for document in source_documents:
    print("---------- meta data ----------")
    print(document.metadata)
    print("------------- text ------------")
    print(document.page_content[:100])

print("------------- answer ------------")
print(result["result"])
