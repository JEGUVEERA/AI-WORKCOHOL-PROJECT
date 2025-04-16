import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import json
import requests
import logging
from io import BytesIO
from dotenv import load_dotenv
from gtts import gTTS
from textblob import TextBlob
from pytrends.request import TrendReq
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from langdetect import detect

# Configure logging for API requests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Generative AI Import
import google.generativeai as genai

# Custom sentiment functions (replacing external dependency for cloud deployment)
def analyze_sentiment_and_emotion(text):
    try:
        # Use TextBlob for basic sentiment analysis
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        # Determine sentiment
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Simplified response
        return f"The text expresses a {sentiment} sentiment with polarity score {polarity:.2f}."
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

def generate_creative_response(text):
    try:
        # Get basic sentiment
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        # Generate creative response using Gemini
        prompt = f"""
        Create a creative and empathetic response to this text: "{text}"
        Make it thoughtful and relevant to the content.
        Keep your response concise (2-3 sentences).
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating creative response: {str(e)}"

def agent(input_data):
    """Simple agent function to replace the external dependency"""
    text = input_data.get("input", "")
    sentiment_analysis = analyze_sentiment_and_emotion(text)
    return {
        "output": f"Based on my analysis: {sentiment_analysis}",
        "sentiment": sentiment_analysis
    }

# Configure model settings
USE_OLLAMA = False  # Set to True if you want to use Ollama

# Get API key from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    # Fallback to .env file for local development
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Ollama (if enabled)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# Check API key
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Please check your Streamlit secrets or .env file.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Configure Streamlit
st.set_page_config(page_title=" AI FOR MARKETING ", layout="wide")
st.sidebar.title("🚀 AI FOR MARKETING ")
# --- Add text-to-speech function
def text_to_speech(text):
    try:
        # Detect language
        language = detect(text)
        tts = gTTS(text=text, lang=language, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# --- Function to generate text content with selected model
def generate_text_content(ingredients: str) -> str:
    prompt = (
        "You are an AI agent for Marketing. "
        "Generate only marketing content (slogans, ad copy, campaign ideas) based on the following input: "
        "no need to generate extra information generate according the prompt. "
        "Do include any extra commentary only related to prompt details and Do not include codes for any content unless I ask for codes. "
        "and real-time information should be generated, AI should be able to generate content based on the user's input and like human. "
        "only marketing content should be generated based on the user's input and according to user's input generate it no need extra information. "
        "and generate only real-time and real-world information based on the user's input. and it should be like created by human not AI. "
        "generate only according to the prompt. "
        "if user given marketing slogans prompt then just generate only marketing slogans prompt information no more information. "
        "if user given ad copy prompt then just generate only ad copy prompt information no more information. "
        "if user given campaign ideas prompt then just generate only campaign ideas prompt information no more information. "
        "if user given slogans prompt then just generate only slogans prompt information no more information. "
        "if user given ideas prompt then just generate only ideas prompt information no more information. "
        "if user given any other prompt then just generate information based on that prompt. "
        "commentary and details should be included. "
        f"{ingredients}."
    )
    
    # Try using Ollama first if enabled
    if USE_OLLAMA:
        try:
            logger.info("Attempting to use Ollama for content generation")
            headers = {"Content-Type": "application/json"}
            data = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
            
            try:
                response = requests.post(OLLAMA_URL, json=data, headers=headers, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response from Ollama")
                else:
                    logger.warning(f"Ollama returned status code {response.status_code}. Falling back to Gemini.")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error connecting to Ollama: {e}. Falling back to Gemini.")
        except Exception as e:
            logger.error(f"Unexpected error with Ollama: {e}. Falling back to Gemini.")
    
    # Fallback to Gemini
    try:
        logger.info("Using Gemini for content generation")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating text content: {e}")
        return "Error generating content."

# --- Create Chat Entry ---
def create_chat_history_entry(feature: str, user_input: str, ai_response: str):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    entry = {"feature": feature, "user": user_input, "ai": ai_response}
    st.session_state.chat_history.append(entry)

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
def generate_multiple_posts(platforms, content, tone="Professional"):
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {}
        for platform in platforms:
            futures[executor.submit(generate_social_media_post, platform, content, tone)] = platform
        for future in as_completed(futures):
            platform = futures[future]
            try:
                results[platform] = future.result()
            except Exception as e:
                results[platform] = f"Error generating post for {platform}: {e}"
    return results





# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Feature Selection ---
page = st.sidebar.radio("Choose a Feature", [
    "Home", "Social Media Post Generator", "Marketing Content Generator",
    "Email Content Generator", "Text Analysis & Sentiment Response", 
    "Text to Speech", "Data Visualization", "Chat History"
])
# --- Home Page ---
if page == "Home":
    st.title("WELCOME TO AI FOR MARKETING")
    st.write("A powerful AI toolset for content generation and analysis.")
    try:
        st.image("AI FOR MARKETING AND AI CONTENT HUB poster.png", 
                caption="AI-Powered Creativity", 
                width=700)
    except:
        st.info("Welcome to the AI for Marketing tool. Choose a feature from the sidebar to get started.")

# --- CHAT HISTORY PAGE ---
elif page == "Data Visualization":
    st.title("📊 AI Data Visualizer")
    st.markdown("Upload your file (CSV, TXT, Excel) or use a sample dataset to visualize numeric data.")
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
        
        # Try using Ollama first if enabled
        if USE_OLLAMA:
            try:
                logger.info(f"Attempting to use Ollama for {platform} post generation")
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
                
                try:
                    response = requests.post(OLLAMA_URL, json=data, headers=headers, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("response", "No response from Ollama")
                    else:
                        logger.warning(f"Ollama returned status code {response.status_code}. Falling back to Gemini.")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error connecting to Ollama: {e}. Falling back to Gemini.")
            except Exception as e:
                logger.error(f"Unexpected error with Ollama: {e}. Falling back to Gemini.")
        
        # Fallback to Gemini
        try:
            logger.info(f"Using Gemini for {platform} post generation")
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
        return generate_text_content(prompt)

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

        # Try using Ollama first if enabled
        if USE_OLLAMA:
            try:
                logger.info("Attempting to use Ollama for email content generation")
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
                
                response = requests.post(OLLAMA_URL, json=data, headers=headers, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response from Ollama")
                else:
                    logger.warning(f"Ollama returned status code {response.status_code}. Falling back to Gemini.")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error connecting to Ollama: {e}. Falling back to Gemini.")
            except Exception as e:
                logger.error(f"Unexpected error with Ollama: {e}. Falling back to Gemini.")
        
        # Fallback to Gemini
        try:
            logger.info("Using Gemini for email content generation")
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
            else:
                logger.error("Received invalid response from Gemini")
                return "Unable to generate email content."
        except Exception as e:
            logger.error(f"Error generating email content with Gemini: {e}")
            st.error(f"Error generating email content: {e}")
            return "Error generating email content."

    # Generate Button
    if st.button("Generate Email"):
        if not email_subject or not email_body:
            st.error("Please enter both the subject and the body to generate an email.")
        else:
            with st.spinner("Generating email content..."):
                try:
                    st.subheader("📩 AI-Generated Email Content")
                    for i in range(num_variants):
                        variant_tone = tone
                        if num_variants > 1:
                            variant_tone = random.choice(["Formal", "Casual", "Playful", "Professional"])
                            st.markdown(f"### ✨ Variant {i+1} ({variant_tone})")

                        email_result = generate_email_content(email_subject, email_body, variant_tone, emoji_boost, include_cta)

                        if email_result and not email_result.startswith("Error"):
                            with st.expander(f"📬 View Variant {i+1}"):
                                st.markdown(email_result)
                                st.code(email_result, language="markdown")

                                st.download_button(
                                    label="📥 Download Email",
                                    data=email_result,
                                    file_name=f"email_variant_{i+1}.txt",
                                    mime="text/plain"
                                )
                    
                    # Save to chat history
                    create_chat_history_entry("Email Generator", f"Subject: {email_subject}", f"Generated {num_variants} email variants")
                    st.success(f"Successfully generated {num_variants} email variant(s)!")
                except Exception as e:
                    st.error(f"An error occurred while generating email content: {e}")
                    logger.error(f"Email generation error: {e}")







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
    if col2.button("Generate Creative Response"):
        if user_input:
            agent_response = agent({"input": user_input})
            creative_response = generate_creative_response(user_input)
            # Display the response in structured format
            st.subheader("🎨 Creative Response")
            st.markdown(f"*Action:* GenerateCreativeResponse")
            st.markdown(f"*Action Input:* {user_input}")
            st.markdown(f"*Generated Response:* {agent_response.get('output', 'No response generated')}")
            st.subheader("✨ Creative Response")
            st.markdown(f"*Generated Creative Response:* {creative_response}")
        else:
            st.warning("Please enter some text for creative response generation.")
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







# --- Text to Speech ---
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
                # Save to history
                create_chat_history_entry("Text to Speech", tts_text, "Audio generated successfully")
            else:
                st.error("Failed to generate audio.")
        else:
            st.warning("Please enter some text.")









# --- Data Visualization ---
elif page == "Data Visualization":
    st.title("📊 AI Data Visualizer")
    st.markdown("Upload your file (CSV, TXT, Excel) or use a sample dataset to visualize numeric data.")

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

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
            st.subheader("⚙️ Chart Configuration")
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
                st.warning("⚠️ Please select at least one column.")
        else:
            st.warning("⚠️ No numeric columns detected in your dataset.")

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
                    st.warning("⚠️ Uploaded file is empty.")
            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")
                data = None

    if data is not None and not data.empty:
        visualize_data(data)
    else:
        st.info("👈 Upload a file or check 'Use sample dataset' to get started.")







if __name__ == "__main__":
    st.write("")
