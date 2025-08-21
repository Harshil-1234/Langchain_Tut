from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

#Langchain doesnot work properly if we declare prompt like below
# chat_template = ChatPromptTemplate([
#     SystemMessage(content = "You are a helpful {domain} expert"),
#     HumanMessage(content="Explain in simple term, what is {topic}")
# ])
chat_template = ChatPromptTemplate([
    ('system',"You are a helpful {domain} expert"),
    ('human',"Explain in simple term, what is {topic}")
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)