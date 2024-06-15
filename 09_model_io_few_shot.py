import os

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms import Ollama

examples = [
    {
        "input": "My favourite foods are apples bananas and grapes",
        "output": "My favourite foods are apples, bananas and grapes.",
    }
]

prompt = PromptTemplate(
    input_variables=["input", "output"], template="input: {input}\noutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="Please fix missing punctuations",
    suffix="input: {input_string}\noutput: ",
    input_variables=["input_string"],
)

llm = Ollama(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

invoked_prompt = few_shot_prompt.invoke(
    {"input_string": "I like the phrase keep it simple stupid"}
)

result = llm.invoke(invoked_prompt.to_messages())

print(invoked_prompt.to_string(), end="")

print(result)
