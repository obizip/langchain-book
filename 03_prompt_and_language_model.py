import os

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

chat = ChatOllama(model="aya", base_url=os.environ["OLLAMA_BASE_URL"])
prompt = PromptTemplate(
    template="Which company invented {product}?", input_variables=["product"]
)

result = chat.invoke([HumanMessage(content=prompt.format(product="BrackBerry"))])

print(result.content)
