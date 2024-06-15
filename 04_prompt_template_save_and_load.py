from langchain.prompts import PromptTemplate, load_prompt

prompt = PromptTemplate(
    template="Which company invented {product}?", input_variables=["product"]
)

prompt_json = prompt.save("prompt.json")
loaded_prompt = load_prompt("prompt.json")

print(loaded_prompt.format(product="BrackBerry"))
