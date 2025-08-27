from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import requests
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a: int, b: int)->int :
    """Given 2 integers : a and b, this tool returns their product/multiplication"""
    return (a*b)

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

model_with_tools = model.bind_tools([multiply])

query = HumanMessage("can you multiply 3 with 20 and tell the result")

messages = [query]

# res = model_with_tools.invoke("can you multiply 3 with 20 and tell the result")
res = model_with_tools.invoke(messages)

messages.append(res)
# print(messages)

res = model_with_tools.invoke(messages)
print(res.content)