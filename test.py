
import os

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI

load_dotenv()

print("API key loaded:", bool(os.getenv("MISTRAL_API_KEY")))

llm = ChatMistralAI(
    model="mistral-small-2603",
    temperature=0
)

response = llm.invoke("Hi")

print(response.content)