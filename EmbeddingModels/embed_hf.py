from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Delhi is the capital of India"

docs = [
    "Delhi is the capital city",
    "Kolkata is the port city",
    "Banglore is the south city"
]

# vector = embedding.embed_query(text)
vector = embedding.embed_documents(docs)

print(vector)
