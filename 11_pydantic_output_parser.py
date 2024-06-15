import os

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

chat = ChatOllama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])


class Smartphone(BaseModel):
    release_date: str = Field(description="release date")
    screen_inches: float = Field(description="screen size(inches)")
    os_installed: str = Field(description="installed OS")
    phone_name: str = Field(description="name of the phone")

    @field_validator("screen_inches")
    def validate_screen_inches(cls, field):
        if field <= 0:
            raise ValueError("Screen inches must be a positive number")
        return field


parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Smartphone), llm=chat
)

result = chat.invoke(
    [
        HumanMessage(content="Tell me a phone released by BlackBerry."),
        HumanMessage(content=parser.get_format_instructions()),
    ]
)

parsed_result = parser.parse(str(result.content))

print(f"name of phone: {parsed_result.phone_name}")
print(f"screen size: {parsed_result.screen_inches}")
print(f"OS: {parsed_result.os_installed}")
print(f"release date: {parsed_result.release_date}")
