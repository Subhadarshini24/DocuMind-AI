from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_conversational_chain():

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    return model