import streamlit as st
import random
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Load Gemini API Key from .env only ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Please check your .env file.")
    st.stop()

# --- Configure Gemini with the loaded API key ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Greeting and Sign-off Helpers ---
def random_greeting():
    return random.choice(["Hi there,", "Hello!", "Greetings,", "Hey!"])

def random_signoff():
    return random.choice(["Cheers", "Best", "Sincerely", "Warm regards"])

# --- Email content generator using Gemini ---
def generate_email_content(subject: str, body: str, tone: str) -> str:
    prompt = (
        f"Generate a {tone.lower()} email with the subject '{subject}' and the following body: '{body}'. "
        "Include a greeting and a closing statement. "
        "Keep the email between 3 to 5 lines. "
        "Write in a natural human tone, not like an AI. "
        "Only marketing content should be generated based on the user's input. "
        "Do not include extra commentary or unnecessary information."
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating email content: {e}")
        return "Error generating email."

# --- Streamlit UI ---
st.title("📧 Email Marketing Content Generator")

tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"])
email_subject = st.text_input("Enter email subject:")
email_body = st.text_area("Enter email body (or key points):")

if st.button("Generate Email"):
    if email_subject and email_body:
        email_result = generate_email_content(email_subject, email_body, tone)
        st.subheader("📩 AI-Generated Email Content")
        st.markdown(email_result)
    else:
        st.error("Please enter both the subject and the body to generate an email.")
