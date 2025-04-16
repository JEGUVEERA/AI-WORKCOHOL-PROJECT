import json
import streamlit as st

# --- Load chat history from file ---
def load_chat_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

## --- Save chat history to file ---
def save_chat_history(chat_history):
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)
    
def display_chat_history(chat_history):
    for message in chat_history:
        st.write(f"{message['role']}: {message['content']}")
    
