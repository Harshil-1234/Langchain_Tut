from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_history = messages=[
    SystemMessage(content='You are a helpful AI assistant'),
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    
    # result = model.invoke(user_input)
    result = model.invoke(chat_history) # using this, model queries on the entire chat history, thus able to understand the context of any upcoming question
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")

print(chat_history)