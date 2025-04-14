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
import pyttsx3import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import json
import requests
import asyncio
import aiohttp
import logging
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from gtts import gTTS
from textblob import TextBlob
from pytrends.request import TrendReq
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import time
import pyttsx3
from gtts import gTTS
import base64
from dotenv import load_dotenv
#from transformers import pipeline
import fal_client

# LangChain & AI Imports

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langdetect import detect



# Custom Modules
from sentiment import agent, analyze_sentiment_and_emotion, generate_creative_response

from chat_utils import load_chat_history, save_chat_history, display_chat_history


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables

if os.getenv("ENV") != "production":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    FAL_KEY = os.getenv("FAL_KEY")
else:
    # Use Streamlit Secrets only in deployment
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    FAL_KEY = st.secrets.get("FAL_KEY", "")

# Check for required API keys
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Please check your .env file or Streamlit secrets.")
    st.stop()


if not FAL_KEY:
    st.error("❌ FAL_KEY is missing. Please check your .env file or Streamlit secrets.")
    st.stop()

os.environ["FAL_KEY"] = FAL_KEY

# --- LangChai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Initialize LangChain LLM ---
llm = OpenAI(temperature=0.7, openai_api_key=GEMINI_API_KEY)

# --- LangChain Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["ingredients"],
    template=(
        "You are an AI agent for Marketing. "
        "Generate only marketing content (slogans, ad copy, campaign ideas, marketing ideas) based on the following input: {ingredients}. "
        "commentary and details should be included."
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
        "commentary and details should be included."
        f"{ingredients}."
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating text content: {e}")
        return "Error generating content."

# Optional offline image generation function

recraft_styles = {
    "Realistic": [
        "realistic_image", "realistic_image/studio_portrait", "realistic_image/natural_light"
    ],
    "Digital Illustration": [
        "digital_illustration/2d_art_poster", "digital_illustration/hand_drawn"
    ],
    "Vector Illustration": [
        "vector_illustration/line_art", "vector_illustration/vivid_shapes"
    ]
}
style_labels = {}
flattened_styles = []
for group, styles in recraft_styles.items():
    for s in styles:
        label = f"{group}: {s.split('/')[-1]}"
        style_labels[label] = s
        flattened_styles.append(label)



def generate_single_image(prompt, style):
    try:
        # Request image generation from the API
        result = fal_client.subscribe(
            "fal-ai/recraft-v3", 
            arguments={
                "prompt": prompt,
                "style": style,
                "image_size": "square_hd"
            },
            with_logs=True
        )
        
        # Log the API response for inspection
        print("API Response:", result)  # Print the full response for debugging
        
        # Check if the response contains the expected image data
        if 'images' in result and len(result['images']) > 0:
            return result['images'][0]['url']  # Return the image URL
        else:
            print("No images returned in the API response.")
            return None  # Return None if no images are found

    except Exception as e:
        # Print the error message to help diagnose issues
        print(f"Error during image generation: {e}")
        return None


def text_to_speech(text):
    tts = gTTS(text)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

def chat_with_ollama(user_input):
    try:
        llm = Ollama(model="llama3.2:1b")
        return llm(user_input)
    except Exception as e:
        return f"Error: {str(e)}"

def generate_multiple_posts(platforms, content):
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_social_media_post, platform, content): platform for platform in platforms}
        for future in as_completed(futures):
            platform = futures[future]
            try:
                results[platform] = future.result()
            except Exception as e:
                results[platform] = f"Error generating post for {platform}: {e}"
    return results


# Configure Streamlit
st.set_page_config(page_title=" AI FOR MARKETING ", layout="wide")
st.sidebar.title("🚀 AI FOR MARKETING ")



# --- AI Model (LangChain) ---
llm = Ollama(model="llama3.2:1b")
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are Jeguveera's AI assistant. "
        "Your job is to generate professional, human-like, real-world content. "
        "Generate high-quality, real-world marketing content. "
        "Only provide responses that sound human-made, persuasive, and relevant to the user's request. "
        "Commentary and details should be included. "
        "Generate complex and detailed content."
    )),
    HumanMessage(content="User query: {query}")
])

chain = prompt | llm | output_parser

# --- Ollama API Query ---

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
headers = {"Content-Type": "application/json"}
data = {"model": "llama3.2:1b", "prompt": "Hello, how are you?", "stream": False}
response = requests.post(OLLAMA_URL, json=data, headers=headers)
print("Ollama API returned:", response.json())

async def query_ollama_async(query):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OLLAMA_URL,
                json={
                    "model": "llama3.2:1b",
                    "prompt": query,
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    return f"Error: Status code {response.status} from Ollama API."

                response_json = await response.json()
                return response_json.get("response", "No response field in Ollama output.")
    except asyncio.TimeoutError:
        return "Error: Request timed out."
    except Exception as e:
        return f"Error: Ollama API connection failed. {str(e)}"

# --- Feature Selection ---
page = st.sidebar.radio("Choose a Feature", [
    "Home", "Chat Bot", "Social Media Post Generator", "Marketing Content Generator",
    "Email Content Generator", "Text Analysis & Sentiment Response", "Image Generator",
    "Text to Speech", "Data Visualization", "Chat History"
])

if page == "Home":
    st.title("WELCOME TO AI FOR MARKETING")
    st.write("A powerful AI toolset for content generation and analysis.")
    st.image("AI FOR MARKETING AND AI CONTENT HUB poster.png", 
             caption="AI-Powered Creativity", 
             width=600)



# --- Chat Bot ---


elif page == "Chat Bot":
    st.title("🤖 Jeguveera's AI Chat Bot")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_ollama_response" not in st.session_state:
        st.session_state.last_ollama_response = ""

    # Chat input with nice UI
    user_input = st.text_input("Type your message here:")
    if user_input:
        st.write("**User:**", user_input)


        with st.spinner("Ollama API thinking..."):
            try:
                ollama_response = asyncio.run(query_ollama_async(user_input))
                st.success("Ollama API 🤖:")
                st.markdown(ollama_response)
            except Exception as e:
                st.error(f"Ollama API Error: {e}")

        # --- Ollama Response ---
        try:
            with st.chat_message("assistant"):
                st.subheader("Ollama  🧠:")
                ollama_response = asyncio.run(query_ollama_async(user_input))

                if "Error: Request timed out." in ollama_response:
                    st.error(ollama_response)
                    if st.session_state.last_ollama_response:
                        st.markdown("Last successful response was:")
                        st.markdown(st.session_state.last_ollama_response)
                else:
                    st.markdown(ollama_response)
                    st.session_state.last_ollama_response = ollama_response
                    st.session_state.chat_history.append({
                        "feature": "Ollama",
                        "user": user_input,
                        "ai": ollama_response
                    })

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Ollama API Error: {e}")

    st.markdown("---")

  




# --- Social Media Post Generator ---


elif page == "Social Media Post Generator":
    st.title("📱 Social Media Post Generator")

    platform_icons = {
        "Twitter": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Logo_of_Twitter.svg/768px-Logo_of_Twitter.svg.png?20220821125553",
        "Instagram": "https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png",
        "Facebook": "https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg",
    }

    def get_trending_topics():
        try:
            pytrends = TrendReq(hl="en-US", tz=360)
            trending_searches_df = pytrends.trending_searches()
            return trending_searches_df[0].tolist()
        except Exception as e:
            return [f"Error fetching trending topics: {e}"]

    def generate_hashtags(content):
        words = content.split()
        keywords = [w.strip("#.,!?") for w in words if len(w) > 3]
        hashtags = [f"#{k.lower()}" for k in keywords[:5]]
        return " ".join(hashtags)

    def generate_social_media_post(platform: str, content: str, tone: str) -> str:
        sentiment = TextBlob(content).sentiment.polarity
        sentiment_label = "neutral"
        if sentiment > 0:
            sentiment_label = "positive"
        elif sentiment < 0:
            sentiment_label = "negative"

        prompt = (
            f"Generate a {platform} post with a {sentiment_label} tone and a {tone} style. "
            f"Content: {content}. Make it engaging and suitable for {platform}. "
            "Add emojis, hashtags, and a clear CTA. Write like a human."
        )
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating social media post: {e}")
            return "Error generating post."

    # --- UI ---
    platform = st.selectbox("Select Platform", list(platform_icons.keys()))
    if platform in platform_icons:
        st.image(platform_icons[platform], width=50)

    tone = st.selectbox("Select Tone", ["Professional", "Funny", "Witty", "Inspirational", "Empathetic"])

    social_content = st.text_area("Write your content idea:", placeholder="Describe what the post should be about...")

    if st.button("✨ Generate Post"):
        if social_content.strip():
            post = generate_social_media_post(platform, social_content, tone)
            hashtags = generate_hashtags(social_content)

            st.subheader("📝 AI-Generated Post")
            st.markdown(post)

            st.subheader("🏷️ Suggested Hashtags")
            st.markdown(hashtags)

   

            # Download Option
            st.download_button("📥 Download Post as Text", data=post, file_name="post.txt")

        else:
            st.error("Please enter some content first.")




   

# --- Marketing Content Generator ---


# --- Marketing Content Generator ---
elif page == "Marketing Content Generator":
    st.title("🚀 Marketing Content Generator")

    tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"], key="tone_selection")
    ingredients = st.text_input("Enter ingredients for marketing content:", placeholder="e.g., marketing slogans, ad copy, campaign ideas")

    if st.button("Generate Marketing Content"):
        if ingredients:
            adjusted_prompt = (
                f"Generate {tone.lower()} marketing content based on the following ingredients: {ingredients}. "
                "Only include content relevant to the prompt.  "
                "Only include content relevant to the prompt. "
                "Be natural and human-like, not robotic or AI-sounding. "
                "Content must reflect real-time, real-world marketing language. "
                "No extra explanation—just the content. "
                "If the prompt is about slogans, only generate slogans. "
                "If it's about ad copy or campaign ideas, generate only that. "
                "If it's something else, follow the prompt faithfully. "
                "It should generate only marketing content."
                "only generate (slogans, ad copy, campaign ideas) based on the following input: {ingredients}. "
                "if only (slogans or  ad copy or campaign ideas)  then only generate the content "
            )

            content = generate_text_content(adjusted_prompt)
            st.subheader("📢 Generated Content")
            st.markdown(content)
            st.download_button("Download Content", content, file_name="marketing_content.txt")

    st.markdown("---")
    st.subheader("🧪 A/B Test Generator")

    marketing_topic = st.text_area("Enter your marketing topic or idea:")

    def generate_marketing_variant(topic):
        styles = ["inspirational", "professional", "fun", "relatable", "bold", "minimal"]
        selected_style = random.choice(styles)
        prompt = (
            f"Create a {selected_style} marketing copy based on the following topic: {topic}. "
            "Make it catchy, compelling, and audience-focused. Keep it concise and effective."
        )
        return generate_text_content(prompt)  # Assuming same function used for text generation

    if st.button("Generate Marketing Variants (A/B Test)"):
        if marketing_topic:
            st.subheader("🎯 A/B Test: Marketing Content Variants")
            for i in range(3):
                st.markdown(f"### 🔹 Variant {i+1}")
                variant = generate_marketing_variant(marketing_topic)
                st.markdown(variant)
        else:
            st.warning("Please enter a marketing topic to generate variants.")





# --- Email Content Generator ---



elif page == "Email Content Generator":
    st.title("📧 Email Marketing Content Generator")

    # Input Fields
    tone = st.selectbox("Select Tone", ["Formal", "Casual", "Playful", "Professional"])
    email_subject = st.text_input("Enter email subject:")
    emoji_boost = st.checkbox("Add emojis to subject (fun & engaging)")
    email_body = st.text_area("Enter email body (or key points):")

    include_cta = st.checkbox("Include a Call-To-Action (CTA) in the email")

    num_variants = st.slider("How many tone variants?", min_value=1, max_value=3, value=1)

    # Email Generation Function
    def generate_email_content(subject: str, body: str, tone: str, emoji: bool, cta: bool) -> str:
        emoji_prefix = "🔥🚀✨💼📣 " if emoji else ""
        subject_text = f"{emoji_prefix}{subject}"

        prompt = (
            f"Generate a {tone.lower()} marketing email with the subject: '{subject_text}'. "
            f"Content should be based on the following key points: '{body}'. "
            "Include a greeting and a closing statement. "
            "Keep the email concise (3 to 5 lines). "
            "Maintain a natural human tone, not robotic. "
        )
        if cta:
            prompt += "Include a clear and compelling Call-To-Action at the end. "
        prompt += "Only generate the email—no extra explanation or metadata."

        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"Error generating email content: {e}")
            return "Error generating email."

    # Generate Button
    if st.button("Generate Email"):
        if email_subject and email_body:
            st.subheader("📩 AI-Generated Email Content")
            for i in range(num_variants):
                variant_tone = tone
                if num_variants > 1:
                    variant_tone = random.choice(["Formal", "Casual", "Playful", "Professional"])
                    st.markdown(f"### ✨ Variant {i+1} ({variant_tone})")

                email_result = generate_email_content(email_subject, email_body, variant_tone, emoji_boost, include_cta)

                with st.expander(f"📬 View Variant {i+1}"):
                    st.markdown(email_result)
                    st.code(email_result, language="markdown")

                    st.download_button(
                        label="📥 Download Email",
                        data=email_result,
                        file_name=f"email_variant_{i+1}.txt",
                        mime="text/plain"
                    )
        else:
            st.error("Please enter both the subject and the body to generate an email.")

# --- Image Generator ---

elif page == "Image Generator":
    st.title("🎨 Realistic AI Image Generator")

    image_prompt = st.text_input("📝 Enter your image prompt:", placeholder="e.g., hyper-realistic portrait of a young astronaut")
    style = st.selectbox("🎨 Choose a Style", ["Realistic", "Anime", "Photographic", "Fantasy", "Digital Art", "No Preference"])

    # Map the selected style to a description
    style_map = {
        "Realistic": "hyper-realistic, photo-like, cinematic lighting",
        "Anime": "anime, cel-shading, high contrast",
        "Photographic": "35mm photo, DSLR, natural light, ultra detailed",
        "Fantasy": "fantasy, magic, ethereal glow, dramatic colors",
        "Digital Art": "digital painting, concept art, stylized, vibrant colors",
        "No Preference": ""
    }

    # Combine the selected style and the user's input prompt
    full_prompt = f"{style_map[style]} {image_prompt}"

    # Generate the image when the button is pressed
    if st.button("Generate Image"):
        if image_prompt.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating image..."):
                # Use 'style' directly in the function instead of 'selected_style'
                image_url = generate_single_image(full_prompt, style)

            if image_url:
                st.image(image_url, caption="🎨 Generated Image", use_column_width=True)
            else:
                st.error("❌ Failed to generate image. Please try again.")
                st.markdown("**Note:** Image generation may take a few moments. Please be patient.")
                st.markdown("**Tip:** Ensure your prompt is clear and descriptive for better results.")
                


# --- Text to Speech ---

elif page == "Text to Speech":
    st.title("🔊 Text to Speech Converter")

    def text_to_speech(text: str, voice: str = "Male") -> str:
        try:
            detected_language = detect(text)
            timestamp = int(time.time())
            filename = f"output_{timestamp}.mp3"

            if detected_language == "en":
                # Use pyttsx3 for English
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')

                # Set voice based on selection
                if voice == "Female":
                    for v in voices:
                        if "female" in v.name.lower() or "zira" in v.name.lower():
                            engine.setProperty('voice', v.id)
                            break
                else:
                    for v in voices:
                        if "male" in v.name.lower():
                            engine.setProperty('voice', v.id)
                            break

                wav_filename = f"output_{timestamp}.wav"
                engine.save_to_file(text, wav_filename)
                engine.runAndWait()
                return wav_filename

            else:
                # Use gTTS for other languages
                tts = gTTS(text=text, lang=detected_language)
                tts.save(filename)
                return filename

        except Exception as e:
            st.error(f"Text-to-Speech Error: {e}")
            return None

    # --- Streamlit UI ---
    tts_text = st.text_area("Enter text to convert to speech:", key="tts_text_area")
    voice = st.selectbox("Select Voice (English only):", ["Male", "Female"])

    if st.button("Convert to Speech"):
        if tts_text:
            audio_file = text_to_speech(tts_text, voice)
            if audio_file and os.path.exists(audio_file):
                st.success("✅ Audio generated successfully!")
                st.audio(audio_file, format="audio/mp3" if audio_file.endswith(".mp3") else "audio/wav")
                st.download_button("Download Audio", open(audio_file, "rb"), file_name=os.path.basename(audio_file))
            else:
                st.error("Failed to generate audio.")
        else:
            st.warning("Please enter text to convert to speech.")




# --- Sentiment Analysis & Creative Response ---
elif page == "Text Analysis & Sentiment Response":
    st.title("📝 Text Analysis and Creative Sentiment Response Generator")
    st.markdown("### Enter your text below:")

    user_input = st.text_area("Input Text:", height=150)

    col1, col2, col3 = st.columns(3)

    # Analyze Sentiment
    if col1.button("Analyze Sentiment"):
        if user_input:
            response = agent({"input": user_input})

            # Display the response in structured format
            st.subheader("📊 Analysis Result")
            st.markdown(f"*Action:* AnalyzeSentiment")
            st.markdown(f"*Action Input:* {user_input}")

            sentiment_and_emotion = analyze_sentiment_and_emotion(user_input)
            st.markdown(f"*Observation:* {sentiment_and_emotion}")
        else:
            st.warning("Please enter some text for sentiment analysis.")

    # Creative AI Response
        # Creative AI Response
    if col2.button("Generate Creative Response"):
        if user_input:
            agent_response = agent.run(user_input)
            creative_response = generate_creative_response(user_input)
            # Display the response in structured format
            st.subheader("🎨 Creative Response")
            st.markdown(f"*Action:* GenerateCreativeResponse")
            st.markdown(f"*Action Input:* {user_input}")
            st.markdown(f"*Generated Response:* {agent_response}")
            st.subheader("✨ Creative Response")
            st.markdown(f"*Generated Creative Response:* {creative_response}")
        else:
            st.warning("Please enter some text for creative response generation.")


    # Agent Debug Trace
    if col3.button("Agent Debug Trace"):
        if user_input:
            # Run agent and sentiment analysis
            response = agent({"input": user_input})
            sentiment = analyze_sentiment_and_emotion(user_input)
            agent_output = response.get("output", "No direct output from agent")
            

        # UI Section
        st.subheader("🤖 Agent Debug Trace")

        with st.expander("🧠 Thought Process"):
            st.markdown("*Thought:* Do I need to use a tool? Yes")
            st.markdown("*Action:* AnalyzeSentiment")
            st.markdown(f"*Action Input:* `{user_input}`")
            st.markdown(f"*Observation:* `{sentiment}`")
            st.markdown("*Thought:* Do I need to use a tool? No")
            st.markdown("*AI Final Response:*")
            st.info(agent_output)

        # Sentiment Badge
        st.markdown("### 🏷️ Sentiment Result")
        sentiment_color = "🟢 Positive" if "positive" in sentiment.lower() else "🔴 Negative" if "negative" in sentiment.lower() else "🟡 Neutral"
        st.success(f"Sentiment Analysis: **{sentiment_color}** — _{sentiment}_")

        


# --- Data Visualization ---



# ------------------------------------------
# ✅ Page freeze fix tips and enhancements:
# ------------------------------------------
# ✅ 1. Check file size & log shape
# ✅ 2. Ensure pandas/numpy are imported
# ✅ 3. Avoid chart overload with too many columns
# ✅ 4. Add fallback and error handling
# ✅ 5. Use caching (optional for large CSVs)
# ✅ 6. Add debug outputs for tracing
# ✅ 7. Provide toggle for sample dataset
# ✅ 8. Prevent full app freeze by limiting selection

# Sidebar Navigation


# --- Data Visualization ---
if page == "Data Visualization":
    st.title("📊 Data Visualization")

    st.markdown("Upload your CSV file or use the sample dataset below.")
    
    # ✅ BONUS: Toggle to use sample dataset
    use_sample = st.checkbox("✅ Use sample dataset instead of uploading")

    data = None  # Initialize data

    # ✅ Optional caching for performance
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)

    # ✅ Load sample or uploaded data
    if use_sample:
        st.info("Using sample dataset (random data).")
        data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Category A', 'Category B', 'Category C']
        )
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                data = load_csv(uploaded_file)
                if not data.empty:
                    st.success("✅ CSV file loaded successfully!")
                    st.write(f"Debug: Data shape: {data.shape}")
                else:
                    st.warning("⚠️ Uploaded CSV is empty.")
                    data = None
            except Exception as e:
                st.error(f"⚠️ Error reading CSV: {e}")
                data = None

    # ✅ If data is loaded properly
    if data is not None and not data.empty:
        st.subheader("📄 Data Preview")
        st.dataframe(data)

        # ✅ Identify numeric columns only
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        st.write("Debug: Numeric columns found:", numeric_cols)

        if numeric_cols:
            st.subheader("📊 Visualization Settings")

            # Chart type selection
            chart_type = st.selectbox("📈 Choose a chart type:", ["Line Chart", "Bar Chart", "Area Chart"])

            # ✅ Limit default selected columns to avoid overload
            default_selection = numeric_cols[:2] if len(numeric_cols) > 2 else numeric_cols
            selected_cols = st.multiselect("🧮 Select columns to visualize:", numeric_cols, default=default_selection)

            st.write("Debug: Selected columns:", selected_cols)

            if not selected_cols:
                st.warning("⚠️ Please select at least one column to visualize.")
            else:
                plot_data = data[selected_cols]

                st.subheader("📊 Chart Output")

                # ✅ Display chart based on user selection
                if chart_type == "Line Chart":
                    st.line_chart(plot_data)
                elif chart_type == "Bar Chart":
                    st.bar_chart(plot_data)
                elif chart_type == "Area Chart":
                    st.area_chart(plot_data)
        else:
            st.warning("⚠️ No numeric columns available to visualize.")
    else:
        st.warning("⚠️ No data available for visualization.")
        st.write("Debug: Data is None or empty after loading.")




# --- Function to Load Chat History ---
def load_chat_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# --- Function to Save Chat History ---
def save_chat_history(history):
    with open("chat_history.json", "w") as f:
        json.dump(history, f)

# --- Function to Display Chat History ---
def display_chat_history():
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.subheader("Chat History:")
        for chat in st.session_state.chat_history:
            st.markdown("---")
            st.markdown(f"**Feature:** {chat['feature']}")
            st.markdown(f"**User:** {chat['user']}")
            st.markdown(f"**AI:** {chat['ai']}")
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No chat history available.")

# --- Function to Create New Entry ---
def create_chat_history_entry(feature: str, user_input: str, ai_response: str):
    st.session_state.chat_history.append({"feature": feature, "user": user_input, "ai": ai_response})
    save_chat_history(st.session_state.chat_history)

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# --- Display Chat History Page ---
if page == "Chat History":
    st.title("Chat History")
    
    if st.button("Show Chat History"):
        display_chat_history()



if __name__ == "__main__":
    # Run the Streamlit app
    st.write("")
from langdetect import detect
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
import logging

logging.basicConfig(level=logging.INFO)
logging.info("App started!!")




# --- Configuration ---


if os.path.exists(".env"):
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
else:
    # Use Streamlit Secrets only in deployment
    # Fallback for local development
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    HF_API_KEY = st.secrets.get("HF_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("Error: GEMINI_API_KEY is missing. Please check your .env file.")
    st.stop()

if not HF_API_KEY:
    st.warning("Warning: HF_API_KEY is missing. Image generation may not work.")

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
  




############################################################################################################

# --- Streamlit UI ---

st.title( "---------- AI FOR MARKETING  ------------------------------------------------------")
st.markdown("*Description:* Text-to-speech functionality in AI Marketing Assistant.")




voice = st.selectbox("Select Voice ", [ "Female"],key="voice_selection")




# --- Text to Speech ---



# --- Function to Split Long Text ---



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

            
            wav_filename = f"output_{timestamp}.wav"
            engine.save_to_file(text, wav_filename)
            engine.runAndWait()

            
            return wav_filename  

        else:
           
            chunks = split_text(text)
            combined_text = " ".join(chunks) 

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
ingredients = st.text_input("Enter ingredients for marketing content:", placeholder="e.g., marketing slogans, slogans, ad copy, campaign ideas , ideas")
if st.button("Generate Marketing Content"):
    if ingredients:
        adjusted_prompt = ("Generate only marketing content (slogans, ad copy, campaign ideas, marketing ideas) based on the following input: {ingredients}."
        "any commentary  only related to prompt details and Do not include codes for any content unless I ask for codes."
        " be like human not AI "
        "Generate only marketing content (slogans, ad copy, campaign ideas, marketing ideas) based on the following input: {ingredients}."
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


st.title("Image Generator  🖼️")

# --- Input Fields ---
image_prompt = st.text_input("📝 Enter your image prompt:", placeholder="e.g., hyper-realistic portrait of a young astronaut")
num_images = st.slider("🖼️ Number of Images", min_value=1, max_value=10, value=1)
guidance_scale = st.slider("🎯 Guidance Scale", 1.0, 20.0, 7.5, step=0.5)
style = st.selectbox("🎨 Choose a Style", ["Realistic", "Anime", "Photographic", "Fantasy", "Digital Art", "No Preference"])

# --- Style Mapping ---
style_map = {
    "Realistic": "hyper-realistic, photo-like, cinematic lighting",
    "Anime": "anime, cel-shading, high contrast",
    "Photographic": "35mm photo, DSLR, natural light, ultra detailed",
    "Fantasy": "fantasy, magic, ethereal glow, dramatic colors",
    "Digital Art": "digital painting, concept art, stylized, vibrant colors",
    "No Preference": ""
}

# --- Image Generation Function ---
def generate_images(prompt: str, num_images: int = 1, style_desc: str = "", guidance: float = 7.5):
    HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    full_prompt = f"{prompt}, {style_desc}".strip(", ")
    images = []

    for i in range(num_images):
        try:
            st.info(f"🧠 Generating Image {i+1}...")
            payload = json.dumps({
                "inputs": full_prompt,
                "options": {
                    "wait_for_model": True,
                    "use_cache": False,
                    "guidance_scale": guidance
                }
            })
            start_time = time.time()
            response = requests.post(api_url, headers=headers, data=payload)
            end_time = time.time()

            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                images.append((img, end_time - start_time))
            else:
                st.error(f"❌ Error generating image {i+1}: {response.content}")
        except Exception as e:
            st.error(f"⚠️ Exception: {e}")
    return images

# --- Generate Button ---
if st.button("🚀 Generate Realistic Images"):
    if image_prompt:
        with st.spinner("Working on it..."):
            images = generate_images(
                prompt=image_prompt,
                num_images=num_images,
                style_desc=style_map.get(style, ""),
                guidance=guidance_scale
            )
        if images:
            st.success("✅ Images generated!")
            cols = st.columns(3)
            for i, (img, duration) in enumerate(images):
                cols[i % 3].image(img, use_container_width=True, caption=f"Image {i+1} (⏱️ {duration:.2f}s)")
                buf = BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label=f"Download Image {i+1}", 
                    data=byte_im,
                    file_name=f"realistic_image_{i+1}.png",
                    mime="image/png"
                )
    else:
        st.warning("⚠️ Please enter an image prompt.")

st.markdown("---")

###############################################################################

# Streamlit UI Setup
st.title("Text Analysis and Creative Sentiment Response Generator  📝 ")
st.markdown("### Enter your text below:")

# User Input
user_input = st.text_area("Input Text:", height=150)

if st.button("Analyze"):
    if user_input:
       
        response = agent({"input": user_input})

      
        st.subheader("Analysis Result:")
        
       
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


def load_chat_history():
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            return json.load(f)
    return []


def save_chat_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()


def create_chat_history_entry(feature: str, user_input: str, ai_response: str):
    st.session_state.chat_history.append({"feature": feature, "user": user_input, "ai": ai_response})
    save_chat_history()  

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
                            data = json.loads(line)  
                            response_text += data.get("response", "") + " "  
                        except json.JSONDecodeError:
                            continue 
                return response_text.strip() if response_text else "Error: No response received"
    except asyncio.TimeoutError:
        return "Error: Request timed out."
    except Exception as e:
        return f"Error: Unable to connect to Ollama API. {str(e)}"

# --- Chatbot UI ---
# User Input
user_input = st.text_input("Enter your message:")

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
      
        ollama_response = asyncio.run(query_ollama_async(user_input))
        if "Error: Request timed out." in ollama_response:
            st.error(ollama_response) 
            if last_ollama_response:  
                st.markdown("Last successful response was:")
                st.markdown(last_ollama_response)  
        else:
            st.markdown(ollama_response)
            last_ollama_response = ollama_response 
            create_chat_history_entry("Jeguveera's Chat Bot", user_input, ollama_response)
    except Exception as e:
        st.error(f"Ollama API Error: {e}")
    st.markdown("---")


# --- Chat History Button ---#

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
        st.markdown("No chat history available.")
