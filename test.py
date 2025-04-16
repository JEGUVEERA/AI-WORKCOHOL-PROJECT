import streamlit as st
import os
import json
#from io import BytesIO  # Missing import for BytesIO
#import requests
#from gtts import gTTS
#from textblob import TextBlob
#from langdetect import detect
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
if os.getenv("ENV") != "production":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
else:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Check API keys
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Please check your .env or Streamlit secrets.")
    st.stop()

# --- LangChain Setup with Gemini API ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- LangChain LLM Setup ---
llm = OpenAI(temperature=0.7, openai_api_key=GEMINI_API_KEY)

# --- LangChain Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["ingredients"],
    template=( 
        "You are an AI agent for Marketing. "
        "Generate only marketing content (slogans, ad copy, campaign ideas, marketing ideas) based on the following input: {ingredients}. "
        "Commentary and details should be included. "
        "Input: {ingredients}\n\n"
        "Output:"
    )
)

# --- LangChain LLM Chain ---
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_text_content(ingredients: str) -> str:
    try:
        response = llm_chain.run(ingredients=ingredients)
        return response
    except Exception as e:
        st.error(f"Error generating text content: {e}")
        return "Error generating content."

# Supported languages by gTTS


# --- Load & Save Chat History ---
def load_chat_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_chat_history(history):
    with open("chat_history.json", "w") as f:
        json.dump(history, f)

# --- Create Chat Entry ---
def create_chat_history_entry(feature: str, user_input: str, ai_response: str):
    entry = {"feature": feature, "user": user_input, "ai": ai_response}
    st.session_state.chat_history.append(entry)
    save_chat_history(st.session_state.chat_history)

# --- Display Chat History Nicely ---
def display_chat_history():
    st.subheader("📜 Previous Conversations")
    chat_history = st.session_state.get("chat_history", [])
    
    if not chat_history:
        st.info("No chat history available.")
    else:
        for i, chat in enumerate(reversed(chat_history), 1):
            st.markdown(f"""
            <div style="border: 1px solid #ff4b4b; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #1e1e1e;">
                <p style="color:#ff4b4b; font-weight:bold;">🔹 Chat #{len(chat_history) - i + 1} - Feature: {chat['feature']}</p>
                <p><strong>User:</strong> {chat['user']}</p>
                <p><strong>AI:</strong> {chat['ai']}</p>
            </div>
            """, unsafe_allow_html=True)

# --- Initialize session_state attributes if they don't exist ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Feature Selection ---
page = st.sidebar.radio("Choose a Feature", [
    "Home", "Chat Bot", "Chat History"
])

# --- Chat Bot Page ---
if page == "Chat Bot":
    st.title("🤖 Jeguveera's AI Chat Bot")
    user_input = st.text_input("Type your message here:")

    if user_input:
        st.write("**User:**", user_input)
        with st.spinner("Gemini API thinking..."):
            try:
                # Use Gemini model to generate a response
                prompt = f"You are a helpful AI assistant. Respond to the following input: {user_input}"
                response = model.generate_content(prompt).text
                st.success("AI Response:")
                st.markdown(response)
                
                # Save to chat history
                create_chat_history_entry("Chat Bot", user_input, response)
            except Exception as e:
                st.error(f"Gemini API Error: {e}")


# --- Home Page ---
elif page == "Home":
    st.title("WELCOME TO AI FOR MARKETING")
    st.write("A powerful AI toolset for content generation and analysis.")
    st.image("AI FOR MARKETING AND AI CONTENT HUB poster.png", 
             caption="AI-Powered Creativity", 
             width=600)

# --- Chat History Page ---
elif page == "Chat History":
    st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="width: 12px; height: 12px; background-color: red; border-radius: 50%; margin-right: 8px;"></div>
            <h3 style="margin: 0; color: white;">Chat History</h3>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Show Chat History"):
        display_chat_history()

    st.markdown("---")
