import json
import requests
import google.generativeai as genai
import os
import streamlit as st
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from gtts import gTTS
import pyttsx3
from langdetect import detect
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import time
import random
from textblob import TextBlob
from pytrends.request import TrendReq



# --- Configuration ---


if os.path.exists(".env"):
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
else:
    # Use Streamlit Secrets only in deployment
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    HF_API_KEY = st.secrets.get("HF_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("Error: GEMINI_API_KEY is missing. Please check your .env file.")
    st.stop()

if not HF_API_KEY:
    st.warning("Warning: HF_API_KEY is missing. Image and video generation may not work.")

# --- Initialize Gemini ---
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
        "Ensure content is real-time, human-like, and relevant. No extra information should be generated."
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
        
        "Do include any extra commentary only related to prompt details and Do not include codes for any content unless I ask for codes."
        "and real-time information should be generated , AI should be able to generate content based on the user's input and like human."
        "only marketing content should be generated based on the user's input and according to user's input generate it no need extra information."
        " and generate only real-time and reai-world information based on the user's input. and it should be like created by human not AI"
        "generate only according to the prompt"
        " if user given marketing slogans prompt then just generate only marketing slogans prompt no more information."
        " if user given ad copy prompt then just generate only ad copy prompt no more information"
        " if user given campaign ideas prompt then just generate only campaign ideas prompt no more information"
        " if user given slogans prompt then just generate only  slogans prompt no more information."
        " if user given ideas prompt then just generate only ideas prompt no more information"
        f"{ingredients}."

    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating text content: {e}")
        return "Error generating content."
  


# --- Concurrent Image Generation Function ---

def generate_images(prompt: str, num_images: int = 4):
    HF_MODEL = "stabilityai/stable-diffusion-2"
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def fetch_image():
        try:
            payload = json.dumps({
                "inputs": prompt,
                "options": {"wait_for_model": True}
            })
            response = requests.post(api_url, headers=headers, data=payload)
            if response.status_code != 200:
                st.error(f"Error generating image: {response.content}")
                return None
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None

    images = []
    max_workers = min(num_images, 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_image) for _ in range(num_images)]
        for future in as_completed(futures):
            img = future.result()
            if img is not None:
                images.append(img)
    return images




# --- Streamlit UI ---

st.title( "--------- AI Marketing Assistant ------------------------------------------------------")
st.markdown("**Description:** Text-to-speech functionality in AI Marketing Assistant.")




voice = st.selectbox("Select Voice", ["Female"], key="voice_selection")



# --- Text to Speech ---
def text_to_speech(text: str, voice: str) -> None:
    try:
        detected_language = detect(text)  # Automatically detect language
        
        if detected_language == "en":
            # Use pyttsx3 for English
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            if voice == "Female":
                for v in voices:
                    if "female" in v.name.lower() or "zira" in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break

            engine.say(text)
            engine.runAndWait()

        else:
            # Use gTTS for other languages
            tts = gTTS(text=text, lang=detected_language)
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")
            os.system("start output.mp3")  # Plays locally

    except Exception as e:
        st.error(f"Error: {e}")

# --- Streamlit UI for Text to Speech ---

st.title("Text to Speech 🎙️")
tts_text = st.text_area("Enter text to convert to speech:", key="tts_text_area")

if st.button("Convert to Speech", key="convert_speech_button"):
    if tts_text:
        text_to_speech(tts_text, voice)
    else:
        st.error("Please enter text to convert to speech.")

st.markdown("---")


######################################################################################

# --- Marketing Content Generation UI ---


st.title("Marketing Content Generator  📢 ")

tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"], key="tone_selection")
ingredients = st.text_input("Enter ingredients for marketing content:", placeholder="e.g., marketing slogans, ad copy, campaign ideas")
if st.button("Generate Marketing Content"):
    if ingredients:
        
        adjusted_prompt = ("Generate {tone.lower()} marketing content based on the following ingredients: {ingredients}"
        "any commentary  only related to prompt details and Do not include codes for any content unless I ask for codes."
        " be like human not AI "
        "and real-time information should be generated , AI should be able to generate content based on the user's input and like human."
        "only marketing content should be generated based on the user's input and according to user's input generate it no need extra information."
        " and generate only real-time and real-world information based on the user's input. and it should be like created by human not AI"
        " no need generate extra information."
        "generate only according to the prompt"
        " if user given marketing slogans prompt then just generate only marketing slogans information no more information."
        " if user given ad copy prompt then just generate only ad copy information no more information"
        " if user given campaign ideas prompt then just generate only campaign ideas information no more information"
        " if in case user given any other prompt then just generate information based on that prompt."
        f"{ingredients}.")
        
        content = generate_text_content(adjusted_prompt)
        st.subheader("Generated Content")
        st.markdown(content)
        st.download_button("Download Content", content, file_name="content.txt")
    else:
        st.error("Please enter ingredients to generate content.")

st.markdown("---")

############################################################################





#######################################################################################################################

# --- Image Generation UI ---


st.title("Image Generator  🖼️ ")
image_prompt = st.text_input("Enter an image prompt:", placeholder="e.g., futuristic ad design")
num_images = st.number_input("Number of Images", min_value=0, max_value=100, value=1)
if st.button("Generate Images"):
    if image_prompt:
        with st.spinner("Generating images..."):
            images = generate_images(image_prompt, num_images=int(num_images))
        if images:
            cols = st.columns(3)
            for i, img in enumerate(images):
                cols[i % 2].image(img, use_container_width=True)
                buf = BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image", 
                    data=byte_im, 
                    file_name=f"generated_image_{i+1}.png", 
                    mime="image/png"
                )
    else:
        st.error("Please enter an image prompt.")

st.markdown("---")


##################################################################





# --- Email Marketing Content Generation ---

def generate_email_content(subject: str, body: str) -> str:
    prompt = (
        f"Generate a {tone.lower()} email with the subject '{subject}' and the following body: '{body}'. "
        "Make sure to include a greeting, a closing statement, and format it as a professional email."
        "and real-time information should be generated , AI should be able to generate content based on the user's input and like human."
        "only marketing content should be generated based on the user's input and according to user's input generate it no need extra information."
        "and no extra information should be generated."
        "body of mail should be 3 to 5 lines"
        "and it should be like created by human not AI"

    )
    try:

        response = model.generate_content(prompt)  # Remove model argument
        return response.text

    except Exception as e:
        st.error(f"Error generating email content: {e}")
        return "Error generating email."

st.title("Email Marketing Content Generator  📧 ")
email_subject = st.text_input("Enter email subject:")
email_body = st.text_area("Enter email body:")

tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"])

if st.button("Generate Email Content"):
    if email_subject and email_body:
        adjusted_email_prompt = f"Generate a {tone.lower()} email with subject '{email_subject}' and body: '{email_body}'"
        email_content = generate_email_content(email_subject, email_body)
        st.subheader("Generated Email Content")
        st.markdown(email_content)
    else:
        st.error("Please enter both subject and body for the email.")


st.markdown("---")




###########################################################################





# --- Social Media Post Generation ---


# --- Social Media Platform Icons ---
platform_icons = {
    "Twitter": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Logo_of_Twitter.svg/768px-Logo_of_Twitter.svg.png?20220821125553",
    "Instagram": "https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png",
    "Facebook": "https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg",
}
def get_trending_topics():
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        trending_searches_df = pytrends.trending_searches()
        return trending_searches_df[0].tolist()  # Returns top trending searches
    except Exception as e:
        return f"Error fetching trending topics: {e}"

# --- Social Media Post Generation ---

def generate_social_media_post(platform: str, content: str) -> str:
    sentiment = TextBlob(content).sentiment.polarity
    sentiment_label = "neutral"
    if sentiment > 0:
        sentiment_label = "positive"
    elif sentiment < 0:
        sentiment_label = "negative"
    
    prompt = (
        f"Generate a {platform} post for the following content: {content}. "
        "Ensure it's engaging, concise, and suitable for the platform. "
        "Include hashtags, emojis, or a call-to-action where appropriate."
        "and real-time information should be generated , AI should be able to generate content based on the user's input and like human."
        "and analyze the generate the information accurately and generate the content."
        " if links are given then just generate the links "
        
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating social media post: {e}")
        return "Error generating post."




st.title(" Social Media Post Generator  📱")
platform = st.selectbox("Select Platform", ["Twitter", "Instagram", "Facebook"])

# Display the selected platform icon
if platform in platform_icons:
    st.image(platform_icons[platform], width=50)  # Display icon

social_content = st.text_area("Enter content for social media post:", placeholder="Write your post content here...")

if st.button("Generate Social Media Post"):
    if social_content:
        post = generate_social_media_post(platform, social_content)
        st.subheader("Generated Social Media Post")
        st.markdown(post)
    else:
        st.error("Please enter content for the social media post.")

st.markdown("---")


    



###############################################################################################################################3333

# --- Chatbot UI ---

st.title(" Jeguveera's Chat Bot  🤖 ")
input_txt = st.text_input("Please enter your queries here...")

# Define Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Your name is Jeguveera's Assistant."
    "generate information based on user's query"
    "information be like created by human not AI"
    "and real-time information should be generated , AI should be able to generate content based on the user's input and like human."
    "and generate in less time"
    "should slove complex probems"
    "if prompt is given related marketing content then just generate marketing content "
    "if prompt is given related social media content then just generate social media content"
    "if prompt is given related any content then just generate that content in professional format"
    "and the output should be in markdown format"
    "and the output should be in less than 2 seconds"
    "and the output should be in less than 2000 characters"),
    ("user", "User query: {query}")
])

# Load LLM
llm = Ollama(model="llama3.2:1b")  # Ensure the model name is correct

output_parser = StrOutputParser()

# Create Chain
chain = prompt | llm | output_parser

if input_txt:
    response = chain.invoke({"query": input_txt})
    st.write(response)  

st.markdown("---")
