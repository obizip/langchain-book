import os

import numpy as np
import numpy.linalg as LA
from langchain_community.embeddings.ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3", base_url=os.environ["OLLAMA_BASE_URL"])

query_vector = embeddings.embed_query("What is the top speed of the flying vehicle?")

print(f"vectorized question: {query_vector[:5]}")

document_1_vector = embeddings.embed_query(
    "The top speed of the flying vehicle is 150 km/h."
)
document_2_vector = embeddings.embed_query(
    "After the chicken has been properly seasoned, it is cooked over medium heat, turning occasionally to ensure that it is fragrant on the outside and tender on the inside."
)

cos_sim_1 = np.dot(query_vector, document_1_vector) / (
    LA.norm(query_vector) * LA.norm(document_1_vector)
)
print(f"Similarity between Document 1 and the question: {cos_sim_1}")

cos_sim_2 = np.dot(query_vector, document_2_vector) / (
    LA.norm(query_vector) * LA.norm(document_2_vector)
)
print(f"Similarity between Document 1 and the question: {cos_sim_2}")
