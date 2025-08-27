from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
import os

load_dotenv()

weather_key = os.getenv("WEATHER_KEY")

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
)

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather(city: str)->str:
    """This function fetches the current weather data for a given city"""

    url = f"http://api.weatherstack.com/current?access_key={weather_key}&query={city}"

    response = requests.get(url)

    return response.json()

# results = search_tool.invoke('top news in India today')

# print(results)

# res = llm.invoke('hi')

# print(res)

prompt = hub.pull('hwchase17/react')
# print(prompt)

agent = create_react_agent(
    llm=llm,
    tools=[search_tool,get_weather],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool,get_weather],
    verbose=True
)

response = agent_executor.invoke({'input':'Find the capital city of Madhya Pradesh, then check its current weather condition'})

print(response)