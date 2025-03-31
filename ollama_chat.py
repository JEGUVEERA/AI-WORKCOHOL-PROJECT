import streamlit as st
import requests

# Define the URL of your local Ollama server
ollama_url = "http://127.0.0.1:11434/api/generate"

# Function to send query to Ollama server
def query_ollama(query):
    response = requests.post(ollama_url, json={"input": query})
    if response.status_code == 200:
        return response.json()['text']
    else:
        return "Error: Unable to get response from Ollama server"

# Streamlit UI setup
st.title("Chat with Ollama AI")
user_input = st.text_input("Enter your message:")

if user_input:
    result = query_ollama(user_input)
    st.write(f"Ollama Response: {result}")
