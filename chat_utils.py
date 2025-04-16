import os
import json

def load_chat_history():
    filepath = "chat_history.json"
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


# --- Save chat history to file ---
def save_chat_history(chat_history):
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)
    
def display_chat_history(chat_history):
    for message in chat_history:
        st.write(f"{message['role']}: {message['content']}")
    
