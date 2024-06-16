import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import RePhraseQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(doc_content_chars_max=500)

chat = ChatOllama(model="codestral", base_url=os.environ["OLLAMA_BASE_URL"])

prompt = PromptTemplate(
    input_variables=["question"],
    template="""Extract the keywords from the following question for a Wikipedia search.
        Question: {question}""",
)

llm_chain = LLMChain(llm=chat, prompt=prompt)

re_phrase_query_retriever = RePhraseQueryRetriever(
    llm_chain=llm_chain, retriever=retriever
)

documents = re_phrase_query_retriever.invoke(
    "I like VIM. By the way, what is bourbon whiskey"
)

print(documents)
