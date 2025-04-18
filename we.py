import os
import re
import json
import streamlit as st
from collections import defaultdict
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel

# ----------------- Load API Key -----------------

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
model = GenerativeModel("gemini-2.0-flash")

# ----------------- Word Lists -----------------
positive_words = [
    "amazing", "wonderful", "fantastic", "excellent", "terrific", "brilliant", "superb",
    "marvelous", "outstanding", "remarkable", "incredible", "phenomenal", "spectacular",
    "extraordinary", "magnificent"
]
negative_words = [
    "infuriating", "agonizing", "unrecoverable", "unethical", "paralyzing", "unsolicited",
    "horrendous", "unsound", "unhelpful", "unresolvable", "failing", "unresolved", "lousy", "regretful"
]
emotion_keywords = {
    "happiness": ["happy", "joy", "excited", "elated", "delighted"],
    "anger": ["angry", "mad", "furious", "irritated", "enraged"],
    "sadness": ["sad", "depressed", "unhappy", "sorrowful", "melancholy"],
    "fear": ["afraid", "scared", "fearful", "terrified", "petrified"],
    "neutral": ["neutral"]
}

# ----------------- Local Sentiment/Emotion Detection -----------------
def detect_emotion_sentiment(text: str) -> dict:
    text_lower = text.lower()
    sentiment_score = 0
    emotion_counts = defaultdict(int)

    for word in positive_words:
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            sentiment_score += 1
    for word in negative_words:
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            sentiment_score -= 1

    for emo, keywords in emotion_keywords.items():
        for word in keywords:
            if re.search(rf"\b{re.escape(word)}\b", text_lower):
                emotion_counts[emo] += 1

    sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1], default=("neutral", 0))[0]

    return {
        "sentiment": sentiment,
        "dominant_emotion": dominant_emotion,
        "emotion_scores": dict(emotion_counts)
    }

# ----------------- Gemini-powered Sentiment/Emotion Detection -----------------
def analyze_with_gemini(text: str) -> dict:
    try:
        prompt = (
            "Analyze the following comment for both **sentiment** and **emotion**:\n\n"
            f"Comment: \"{text}\"\n\n"
            "Respond in JSON format like:\n"
            "{\n"
            "  \"sentiment\": \"Positive | Negative | Neutral\",\n"
            "  \"emotions\": [\"happiness\", \"anger\"]\n"
            "}"
        )
        response = model.generate_content(prompt)
        cleaned_text = response.text.strip("` \n")

        try:
            result = json.loads(cleaned_text)
        except:
            result = {"sentiment": "Unknown", "emotions": []}
            for line in cleaned_text.splitlines():
                if "sentiment" in line.lower():
                    if "positive" in line.lower():
                        result["sentiment"] = "Positive"
                    elif "negative" in line.lower():
                        result["sentiment"] = "Negative"
                    elif "neutral" in line.lower():
                        result["sentiment"] = "Neutral"
                if "emotion" in line.lower():
                    emotions = line.split(":")[-1].strip()
                    result["emotions"] = [e.strip().capitalize() for e in emotions.split(",") if e.strip()]
        return result

    except Exception as e:
        return {"sentiment": "Error", "emotions": []}

# ----------------- Creative Response Generator -----------------
def generate_creative_response(text: str) -> str:
    return f"✨ Here's a creative twist: {text[::-1]}"

# ----------------- Streamlit UI -----------------

page = st.sidebar.radio("Choose a Feature", [
    "Text Analysis and Creative Response"
])

if page == "Text Analysis and Creative Response":
    st.title("📝 Text Analysis and Creative Sentiment Response")
    st.markdown("Enter some text and analyze its *sentiment and emotion*, or generate a creative response.")

    user_input = st.text_area("Input Text", height=150)
    fast_mode = st.checkbox("⚡ Fast Mode (Local Rule-based Analysis)", value=True)

    col1, col2 = st.columns(2)
    analyze_btn = col1.button("🔍 Analyze Sentiment & Emotion")
    creative_btn = col2.button("🎨 Generate Creative Response")

    if user_input.strip():
        if analyze_btn:
            st.subheader("📊 Sentiment & Emotion Analysis")
            if fast_mode:
                result = detect_emotion_sentiment(user_input)
            else:
                result = analyze_with_gemini(user_input)

            st.markdown(f"- **Sentiment:** {result.get('sentiment', 'N/A')}")
            if 'dominant_emotion' in result:
                st.markdown(f"- **Dominant Emotion:** {result['dominant_emotion']}")
            if 'emotions' in result:
                st.markdown(f"- **Emotions:** {', '.join(result['emotions'])}")
            if 'emotion_scores' in result:
                st.markdown(f"- **Emotion Scores:** {result['emotion_scores']}")

        if creative_btn:
            st.subheader("🎨 Creative Response")
            st.markdown(generate_creative_response(user_input))
    else:
        st.info("💬 Please enter some text above to get started.")
