from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Which company invented {product}?", input_variables=["product"]
)

print(prompt.format(product="BrackBerry"))
print(prompt.format(product="Nothing Phone"))
