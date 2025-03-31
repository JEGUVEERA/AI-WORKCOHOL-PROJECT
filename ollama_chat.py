import streamlit as st
import requests
import json  # Import JSON module to parse streaming response

# Define Ollama API URL
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Function to send query to Ollama server and process streaming response
def query_ollama(query):
    response = requests.post(OLLAMA_URL, json={"model": "llama3.2:1b", "prompt": query}, stream=True)
    
    response_text = ""
    
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)  # Parse each JSON line
                response_text += data.get("response", "") + " "  # Extract response text
            except json.JSONDecodeError:
                continue  # Skip lines that aren't valid JSON
    
    return response_text.strip() if response_text else "Error: No response received"

# Streamlit UI setup
st.title("Jeguveera's Chat Bot 🤖")
user_input = st.text_input("Enter your message:")

if user_input:
    result = query_ollama(user_input)
    st.write(f"Ollama Response: {result}")
