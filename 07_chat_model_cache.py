import os
import time

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

set_llm_cache(InMemoryCache())

chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

start = time.time()
result = chat.invoke([HumanMessage(content="Hi!")])
end = time.time()

print(result.content)
print(f"time: {end - start} sec\n")

start = time.time()
result = chat.invoke([HumanMessage(content="Hi!")])
end = time.time()

print(result.content)
print(f"time: {end - start} sec\n")
