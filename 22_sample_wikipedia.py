from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)

documents = retriever.invoke("Large Language Model")

print(f"search results: {len(documents)}")

for document in documents:
    print("---------- meta data ----------")
    print(document.metadata)
    print("------------- text ------------")
    print(document.page_content[:100])
