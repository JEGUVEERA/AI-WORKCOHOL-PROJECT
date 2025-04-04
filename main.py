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
import shutil

import random
from textblob import TextBlob
from pytrends.request import TrendReq
import asyncio
import aiohttp
from sentiment import agent, analyze_sentiment






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


############################################################################################################

# --- Streamlit UI ---

st.title( "--------- AI Marketing Assistant ------------------------------------------------------")
st.markdown("*Description:* Text-to-speech functionality in AI Marketing Assistant.")




voice = st.selectbox("Select Voice ", [ "Female"],key="voice_selection")




# --- Text to Speech ---



# --- Function to Split Long Text ---


# --- Optional: Split for extreme cases, but we won’t save multiple files
def split_text(text, max_chars=300):
    if len(text) <= max_chars:
        return [text]
    
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            current_chunk += " " + word

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# --- Text-to-Speech Function ---
def text_to_speech(text: str, voice: str) -> str:
    try:
        detected_language = detect(text)
        timestamp = int(time.time())
        filename = f"output_{timestamp}.mp3"

        if detected_language == "en":
            # Use pyttsx3 for English
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            if voice == "Female":
                for v in voices:
                    if "female" in v.name.lower() or "zira" in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break

            # Save as WAV (pyttsx3 doesn't support MP3 directly)
            wav_filename = f"output_{timestamp}.wav"
            engine.save_to_file(text, wav_filename)
            engine.runAndWait()

            # Optional: Convert WAV to MP3 (if needed), or just use WAV
            return wav_filename  # Return the file path

        else:
            # Use gTTS for non-English and save as one MP3
            chunks = split_text(text)
            combined_text = " ".join(chunks)  # Merge all chunks into one string

            tts = gTTS(text=combined_text, lang=detected_language)
            tts.save(filename)
            return filename

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- Streamlit UI ---
st.title("Text to Speech 🎙")
tts_text = st.text_area("Enter text to convert to speech:", key="tts_text_area")
voice = st.selectbox("Select Voice (Only for English)", ["Male", "Female"])

if st.button("Convert to Speech", key="convert_speech_button"):
    if tts_text:
        audio_file = text_to_speech(tts_text, voice)
        if audio_file and os.path.exists(audio_file):
            st.success("✅ Audio generated successfully!")
            st.audio(audio_file, format="audio/mp3" if audio_file.endswith(".mp3") else "audio/wav")
            st.download_button("Download Audio", open(audio_file, "rb"), file_name=os.path.basename(audio_file))
            os.system(f'start {audio_file}')  # Optional: play it locally
    else:
        st.error("Please enter text to convert to speech.")


######################################################################################

# --- Marketing Content Generation UI ---

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
        " if user given marketing slogans prompt then just generate only marketing slogans prompt information no more information."
        " if user given ad copy prompt then just generate only ad copy prompt information no more information"
        " if user given campaign ideas prompt then just generate only campaign ideas prompt information no more information"
        " if in case user given any other prompt then just generate information based on that prompt."
        f"{ingredients}.")
        
        content = generate_text_content(adjusted_prompt)
        st.subheader("Generated Content")
        st.markdown(content)
        st.download_button("Download Content", content, file_name="content.txt")
    
st.markdown("---")




############################################################################



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

###############################################################################

# Streamlit UI Setup
st.title("Text Analysis and Creative Sentiment Response Generator  📝 ")
st.markdown("### Enter your text below:")

# User Input
user_input = st.text_area("Input Text:", height=150)

if st.button("Analyze"):
    if user_input:
        # Invoke the agent with the user input
        response = agent({"input": user_input})

        # Display the response in the structured format
        st.subheader("Analysis Result:")
        
        # Start of the structured response
        st.markdown("*Thought:* Do I need to use a tool? Yes")
        st.markdown(f"*Action:* AnalyzeSentiment")
        st.markdown(f"*Action Input:* {user_input}")
        
        sentiment = analyze_sentiment(user_input)
        st.markdown(f"*Observation:* {sentiment}")
        
        st.markdown("*Thought:* Do I need to use a tool? No")
        st.markdown("*AI:* I understand your frustration. Could you tell me more about what specifically didn't work for you? Knowing more details might help others avoid a similar disappointment.")

    else:
        st.error("Please enter some text to analyze.")

st.markdown("---")





#######################################################################################################################



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
tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"])
email_subject = st.text_input("Enter email subject:")
email_body = st.text_area("Enter email body:")

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


# Streamlit UI Setup
st.title("Jeguveera's Chat Bot 🤖")
st.markdown("### Your AI Assistant for Generating Professional Content")

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


# --- Chat History Button ---#
# Optional: Show chat history if needed
if st.button("Show Chat History"):
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.subheader("Chat History:")
        
        for chat in st.session_state.chat_history:
            st.markdown("---")
            st.markdown(f"*Feature:* {chat['feature']}")
            st.markdown(f"*User:* {chat['user']}")
            st.markdown(f"*AI:* {chat['ai']}")
            st.markdown("<br>", unsafe_allow_html=True)  # Add a line break for spacing
            st.markdown("---")  # Optional separator line
    else:
        