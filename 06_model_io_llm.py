import os

from langchain_community.llms import Ollama

llm = Ollama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

result = llm.invoke("The best OS is", stop=["."])

print(result)
