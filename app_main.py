import streamlit as st
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
from dotenv import load_dotenv
from gtts import gTTS
from textblob import TextBlob
from pytrends.request import TrendReq
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import time
from dotenv import load_dotenv
from langdetect import detect





# LangChain & AI Imports

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
#from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langdetect import detect
from langchain_openai import OpenAI





# Custom Modules
from sentiment import agent, analyze_sentiment_and_emotion, generate_poetic_response


from chat_utils import load_chat_history, save_chat_history, display_chat_history





# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables

if os.getenv("ENV") != "production":
    load_dotenv()
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Check if the keys are set
else:
    
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    
# Check API keys

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Please check your .env or Streamlit secrets.")
    st.stop()






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



# Supported languages by gTTS
SUPPORTED_LANGUAGES = {
    "af", "ar", "bn", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "eo", "es", "et", "fi", "fr", "gu", "hi", "hr",
    "hu", "id", "is", "it", "ja", "jw", "kn", "ko", "la", "lv", "mk", "ml", "mr", "my", "ne", "nl", "no", "pl", "pt",
    "ro", "ru", "si", "sk", "sq", "sr", "su", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "zh-CN", "zh-TW", "zh"
}

def text_to_speech(text: str) -> BytesIO:
    try:
        detected_language = detect(text)
        if detected_language not in SUPPORTED_LANGUAGES:
            st.warning(f"⚠ Detected language '{detected_language}' is not supported. Using English instead.")
            detected_language = "en"

        tts = gTTS(text=text, lang=detected_language)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer

    except Exception as e:
        st.error(f"Text-to-Speech Error: {e}")
        return None



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

# --- Async Query Function (Placeholder) ---
async def query_ollama_async(prompt):
    await asyncio.sleep(1)  # Simulate API delay
    return f"Response to: {prompt}"


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









### --- Feature Selection ---
page = st.sidebar.radio("Choose a Feature", [
    "Home", "Chat Bot", "Social Media Post Generator", "Marketing Content Generator",
    "Email Content Generator", "Text Analysis & Sentiment Response", 
    "Text to Speech", "Data Visualization", "Chat History"
])

if page == "Home":
    st.title("WELCOME TO AI FOR MARKETING")
    st.write("A powerful AI toolset for content generation and analysis.")
    st.image("AI FOR MARKETING AND AI CONTENT HUB poster.png", 
             caption="AI-Powered Creativity", 
             width=600)






# --- Chat Bot ---

# --- SESSION STATE INIT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if "last_ollama_response" not in st.session_state:
    st.session_state.last_ollama_response = ""

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



# --- CHAT HISTORY PAGE ---
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

            st.subheader("🏷 Suggested Hashtags")
            st.markdown(hashtags)

   

            # Download Option
            st.download_button("📥 Download Post as Text", data=post, file_name="post.txt")

        else:
            st.error("Please enter some content first.")







   



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



# --- Sentiment Analysis & Creative Response ---
elif page == "Text Analysis & Sentiment Response":
    st.title("📝 Text Analysis and Creative Sentiment Response Generator")
    st.markdown("### Enter your text below:")

    user_input = st.text_area("Input Text:", height=150)

    col1, col2, col3 = st.columns(3)

# Button 1: Analyze Sentiment
if col1.button("Analyze Sentiment"):
    if user_input:
        result = analyze_sentiment_and_emotion(user_input)
        st.subheader("📊 Sentiment Analysis")
        st.markdown(f"- **Sentiment:** `{result['sentiment']}`")
        st.markdown(f"- **Dominant Emotion:** `{result['emotion']}`")
        st.markdown("#### Emotion Word Matches:")
        st.json(result["counts"])
    else:
        st.warning("Please enter text to analyze.")

# Button 2: Generate Poetic Response
if col2.button("Generate Creative Response"):
    if user_input:
        poetic = generate_poetic_response(user_input)
        st.subheader("🎨 Poetic Response")
        st.markdown(poetic)
    else:
        st.warning("Please enter text for creative response.")

# Button 3: Full Agent Debug
if col3.button("Agent Debug"):
    if user_input:
        response = agent({"input": user_input})
        analysis = analyze_sentiment_and_emotion(user_input)
        st.subheader("🤖 Agent Debug Info")
        st.markdown(f"**Input Text:** {user_input}")
        st.markdown(f"**Agent Output:** {response.get('output', 'N/A')}")
        st.markdown(f"**Sentiment:** {analysis['sentiment']} | **Emotion:** {analysis['emotion']}")
        st.markdown("**Emotion Counts:**")
        st.json(analysis["counts"])
    else:
        st.warning("Please input text for agent debug trace.")





# --- Text to Speech ---#

# --- UI: Text to Speech Page ---
elif page == "Text to Speech":
    st.title("🔊 Text to Speech Converter (Multilingual)")

    tts_text = st.text_area("Enter text to convert to speech (supports many languages):", key="tts_text_area")

    if st.button("Convert to Speech"):
        if tts_text.strip():
            audio_buffer = text_to_speech(tts_text)
            if audio_buffer:
                st.success("✅ Audio generated successfully!")
                st.audio(audio_buffer, format="audio/mp3")
                st.download_button("Download Audio", audio_buffer, file_name=f"tts_output_{int(time.time())}.mp3")
            else:
                st.error("Failed to generate audio.")
        else:
            st.warning("Please enter some text.")









# --- Data Visualization ---
elif page == "Data Visualization":
    st.title("📊 AI Data Visualizer")
    st.markdown("Upload your file (CSV, TXT, Excel) or use a sample dataset to visualize numeric data.")


    def load_file(file):
        ext = os.path.splitext(file.name)[-1]
        try:
            if ext in [".csv", ".txt"]:
                sample = file.read(2048).decode("ISO-8859-1")
                file.seek(0)
                delimiter = "," if sample.count(",") > sample.count("\t") else "\t"
                df = pd.read_csv(file, encoding="ISO-8859-1", delimiter=delimiter, on_bad_lines="skip")
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file)
            else:
                return None

            # Clean numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")[0],
                    errors="coerce"
                )

            return df
        except Exception as e:
            raise e

    def plot_custom(data, selected_cols):
        st.markdown("### 📅 Time Series / Custom Line Plot")
        fig, ax = plt.subplots(figsize=(10, 6))

        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            for col in selected_cols:
                ax.plot(data["Date"], data[col], label=col)
            ax.set_xlabel("Date")
        else:
            for col in selected_cols:
                ax.plot(data.index, data[col], label=col)
            ax.set_xlabel("Index")

        ax.set_ylabel("Value")
        ax.set_title("Custom Plot")
        ax.legend()
        st.pyplot(fig)

    def visualize_data(data):
        st.subheader("📄 Data Preview")
        st.dataframe(data)
        data = data.reset_index(drop=True)

        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            st.subheader("⚙ Chart Configuration")
            chart_type = st.selectbox("📈 Select chart type:", ["Line Chart", "Bar Chart", "Area Chart", "Custom Plot"])
            selected_cols = st.multiselect("📌 Select numeric columns to visualize:", numeric_cols, default=numeric_cols[:2])

            if selected_cols:
                st.subheader("📊 Chart Output")
                plot_data = data[selected_cols]

                if chart_type == "Line Chart":
                    st.line_chart(plot_data)
                elif chart_type == "Bar Chart":
                    st.bar_chart(plot_data)
                elif chart_type == "Area Chart":
                    st.area_chart(plot_data)
                elif chart_type == "Custom Plot":
                    plot_custom(data, selected_cols)
            else:
                st.warning("⚠ Please select at least one column.")
        else:
            st.warning("⚠ No numeric columns detected in your dataset.")

    # Upload or use sample
    use_sample = st.checkbox("✅ Use sample dataset instead of uploading")
    data = None

    if use_sample:
        st.info("Using sample dataset (random values).")
        data = pd.DataFrame(np.random.randn(20, 3), columns=['Category A', 'Category B', 'Category C'])
    else:
        uploaded_file = st.file_uploader("📁 Upload a file", type=["csv", "txt", "xlsx", "xls"])
        if uploaded_file:
            try:
                data = load_file(uploaded_file)
                if data is not None and not data.empty:
                    st.success("✅ File loaded successfully!")
                else:
                    st.warning("⚠ Uploaded file is empty.")
            except Exception as e:
                st.error(f"⚠ Error reading file: {e}")
                data = None

    if data is not None and not data.empty:
        visualize_data(data)
    else:
        st.info("👈 Upload a file or check 'Use sample dataset' to get started.")







if __name__ == "_main_":
    st.write("")
