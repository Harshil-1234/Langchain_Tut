from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

userInput = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = model.invoke(userInput)
    print(result)
    st.write(result.content)

