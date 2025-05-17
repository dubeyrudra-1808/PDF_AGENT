import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Please set your GROQ_API_KEY in the .env file")
        return

    llm = ChatGroq(
        model_name="llama3-8b-8192",
        groq_api_key=groq_api_key
    )

    prompt = "Explain Artificial Intelligence in simple terms."

    # Create a list of messages (HumanMessage)
    messages = [HumanMessage(content=prompt)]

    # Use invoke() instead of __call__
    response = llm.invoke(messages)

    print("Groq response:", response.content)

if __name__ == "__main__":
    test_groq()
