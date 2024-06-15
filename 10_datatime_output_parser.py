import os

from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

output_parser = DatetimeOutputParser()

chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

prompt = PromptTemplate.from_template("What is the release date of the {product}?")

result = chat.invoke(
    [
        HumanMessage(content=prompt.format(product="BlackBerry")),
        HumanMessage(content=output_parser.get_format_instructions()),
    ]
)
output = output_parser.parse(str(result.content))

print(output)
