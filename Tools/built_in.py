from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

search_tool = DuckDuckGoSearchRun()

prompt = PromptTemplate(
    template="Summarize the below content in 1 or 2 lines explaining the text to user\n{content}",
    input_variables=['content']
)

chain = search_tool | parser | prompt | model | parser

# results = chain.invoke('search for the weather in delhi today')
results2 = search_tool.invoke('weather in delhi today')

print(results2)