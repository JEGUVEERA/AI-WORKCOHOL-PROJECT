import os
import json
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# --- Load Gemini API Key from .env only ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Please check your .env file.")
    st.stop()

# --- Configure Gemini with the loaded API key ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Initialize LangChain LLM ---
llm = OpenAI(temperature=0.7, openai_api_key=GEMINI_API_KEY)

# --- LangChain Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["ingredients"],
    template=(
        "You are an AI agent for Marketing. "
        "Generate only marketing content (slogans, ad copy, campaign ideas) based on the following input: {ingredients}. "
        
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



def generate_text_content(ingredients: str) -> str:
    prompt = (
        "You are an AI agent for Marketing. "
        "Generate only marketing content (slogans, ad copy, campaign ideas) based on the following input: "
        "no need to generate extra information generate according the prompt"
        "Do include any extra commentary only related to prompt details and Do not include codes for any content unless I ask for codes."
        "and real-time information should be generated , AI should be able to generate content based on the user's input and like human."
        "only marketing content should be generated based on the user's input and according to user's input generate it no need extra information."
        " and generate only real-time and reai-world information based on the user's input. and it should be like created by human not AI"
        "generate only according to the prompt"
        " if user given marketing slogans prompt then just generate only marketing slogans prompt information no more information."
        " if user given ad copy prompt then just generate only ad copy prompt information no more information"
        " if user given campaign ideas prompt then just generate only campaign ideas prompt information no more information"
        " if user given slogans prompt then just generate only  slogans prompt information no more information."
        " if user given ideas prompt then just generate only ideas prompt information no more information"
        "if user given any other prompt then just generate information based on that prompt."
        f"{ingredients}."

    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating text content: {e}")
        return "Error generating content."


# 🔹 Marketing Content Generation using Gemini
def generate_marketing_content(product, audience, features, slogans=None, ad_copy=None, campaign_ideas=None):
    try:
        prompt = f"Create persuasive marketing content for a product called '{product}' targeting {audience}. Highlight these features: {features}."

        if slogans:
            prompt += f" Include slogans like: {slogans}."
        if ad_copy:
            prompt += f" Use ad copy style like: {ad_copy}."
        if campaign_ideas:
            prompt += f" Consider campaign ideas like: {campaign_ideas}."

        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        return f"❌ Error generating marketing content: {e}"
