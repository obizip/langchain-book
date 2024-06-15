import os

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

for chunk in chat.stream(
    [HumanMessage(content="Tell me the best way to create a python environment?")]
):
    print(chunk.content, end="", flush=True)
