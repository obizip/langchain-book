import os

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

result = chat.invoke([HumanMessage(content="Hello!")])
print(result.content)
