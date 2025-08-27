from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# query = "Tell me about Virat Kohli "
query = "Tell me about Bumrah "

#Here we are making model generate embeddings continuously which is expensive, so we store them in vector db on first time generation
vector_doc = embedding.embed_documents(documents)
vector_query = embedding.embed_query(query)

scores = cosine_similarity([vector_query],vector_doc)[0]

index,score = (sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])

print("User: ",query)

print(documents[index])

print("Similarity score is: ",score)