from langchain.chains import LLMChain
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser

import os
import streamlit as st
import json
import aiohttp
import asyncio
# other imports...


llm = Ollama(model="llama3.2:1b")

def chatbot_response(question: str) -> str:
    prompt = PromptTemplate.from_template("Q: {question} A:")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question)


## --- Function to Load Chat History ---
def load_chat_history():
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            return json.load(f)
    return []

# --- Function to Save Chat History ---
def save_chat_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f)

# Ensure session state for chat history exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# --- Function to create chat history entry ---
def create_chat_history_entry(feature: str, user_input: str, ai_response: str):
    st.session_state.chat_history.append({"feature": feature, "user": user_input, "ai": ai_response})
    save_chat_history()  # Save the history to file after adding a new entry

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Jeguveera's AI assistant, designed to generate professional, real-world content."
               "Generate responses that appear to be written by a human."
               "You must solve complex problems, provide real-time insights, and generate responses within 2 seconds."
               "If the query is related to marketing, social media, or any specific content, generate professional content accordingly."
               "The response should always be in markdown format and under 2000 characters."
               "Additionally, translate content into multiple languages based on user input."
               "no need generate any code unless asked by user"
               "if any promt is given generate  that with high accuracy"
               "and generate  responses within 2 seconds in professional way"
               "add translate bot feature to translate content into multiple languages based on user input if needed in case then only have to  generate other wise no need"
               "in langchain output  should generate all responses which is related to given prompt"),
    ("user", "User query: {query}")
])

# Load LLM
llm = Ollama(model="llama3.2:1b")

# Output Parser
output_parser = StrOutputParser()

# Create LangChain Processing Chain
chain = prompt | llm | output_parser

# Ollama API URL
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Function to Query Ollama API (Streaming Response)
async def query_ollama_async(query):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_URL, json={"model": "llama3.2:1b", "prompt": query}) as response:
                if response.status != 200:
                    return f"Error: Received status code {response.status} from Ollama API."

                response_text = ""
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)  # Parse each JSON line
                            response_text += data.get("response", "") + " "  # Extract response text
                        except json.JSONDecodeError:
                            continue  # Skip lines that aren't valid JSON
                return response_text.strip() if response_text else "Error: No response received"
    except asyncio.TimeoutError:
        return "Error: Request timed out."
    except Exception as e:
        return f"Error: Unable to connect to Ollama API. {str(e)}"

# --- Chatbot UI ---
# User Input
user_input = st.text_input("Enter your message:")

# Variable to store the last successful Ollama response
last_ollama_response = ""

st.markdown("---")

if user_input:
    st.subheader("LangChain Output:")
    try:
        response = chain.invoke({"query": user_input})
        st.markdown(response)
        create_chat_history_entry("Jeguveera's Chat Bot", user_input, response)
    except Exception as e:
        st.error(f"LangChain Error: {e}")

    st.subheader("Ollama API Output:")
    try:
        # Call the asynchronous function
        ollama_response = asyncio.run(query_ollama_async(user_input))
        if "Error: Request timed out." in ollama_response:
            st.error(ollama_response)  # Display the timeout error
            if last_ollama_response:  # Check if there is a last successful response
                st.markdown("Last successful response was:")
                st.markdown(last_ollama_response)  # Show the last successful response
        else:
            st.markdown(ollama_response)
            last_ollama_response = ollama_response  # Update the last successful response
            create_chat_history_entry("Jeguveera's Chat Bot", user_input, ollama_response)
    except Exception as e:
        st.error(f"Ollama API Error: {e}")
    st.markdown("---")

