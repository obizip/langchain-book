import os

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

output_parser = CommaSeparatedListOutputParser()

# llama3 didn't follow instructions haha
chat = ChatOllama(model="aya", base_url=os.environ["OLLAMA_BASE_URL"])

result = chat.invoke(
    [
        HumanMessage(content="What are three major products developed by IBM?"),
        HumanMessage(content=output_parser.get_format_instructions()),
    ]
)
output = output_parser.parse(str(result.content))

for item in output:
    print("major product => " + item)
