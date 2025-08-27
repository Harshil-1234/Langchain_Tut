#Tool 1 -> fetch conversion rate between currencies
#Tool 2 -> convert the currencies

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import requests
from langchain_core.tools import tool,InjectedToolArg
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Annotated
from dotenv import load_dotenv
import os

load_dotenv()
conversion_key = os.getenv("EXCHANGE_RATE_API_KEY")

@tool
def getConversionFactor(base_currency: str, target_currency: str)->float :
    """
    This function fetches the currency conversion factor between the base currency and target currency
    """
    url = f'https://v6.exchangerate-api.com/v6/{conversion_key}/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

#This tells LLM not to fill the conversion_rate argument, the developer will inject based on prev tool's val.
@tool
def convertCurrency(base_currency_value: int, conversion_rate: Annotated[float,InjectedToolArg])->float :
    """
    given a currency conversion rate this function calculates the target currency value from a given base   currency value
    """
    return (base_currency_value * conversion_rate)

# res = getConversionFactor.invoke({'base_currency':'USD','target_currency':'INR'})
# res = convertCurrency.invoke({'base_currency_value':10,'conversion_rate':'81.16'})

llm = HuggingFaceEndpoint(
    # repo_id="google/gemma-2-2b-it",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Call tool manually
conversion = getConversionFactor.invoke({
    "base_currency": "USD",
    "target_currency": "INR"
})
rate = conversion["conversion_rate"]  # assuming your API returns this key
converted = convertCurrency.invoke({
    "base_currency_value": 10,
    "conversion_rate": rate
})

# Let the LLM summarize it
prompt = (
    f"""The conversion rate from USD to INR is {rate}.
    So, 10 USD equals {converted:.2f} INR. "
    Explain this to a user in friendly tone.""",
)

# LLM will turn it into a nice response
response = model.invoke(prompt)
print(response.content)

