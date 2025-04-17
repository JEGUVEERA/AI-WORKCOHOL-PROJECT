import re
import os
from dotenv import load_dotenv
from collections import defaultdict
from langchain.tools import Tool  
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI




load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)


positive_words = [
    "happy", "joy", "love", "great", "amazing", "wonderful", "brilliant", "cheerful", "delightful", "ecstatic",
    
    "fantastic", "grateful", "harmonious", "inspiring", "jubilant", "kind", "lively", "marvelous", "optimistic",
]

negative_words = [
    "sad", "angry", "hate", "bad", "awful", "terrible", "horrible", "miserable", "depressed", "annoyed",
   "unhelpful", "unresolvable", "failing", "unresolved", "lousy", "regretful"
]

# Emotion mapping
emotion = {
    "joy": 
    ["happy", "joy", "delighted", "cheerful", "smile", "grateful", "optimistic", "excited","happy", "joy", "delighted", "cheerful", "content", "service", "duty", "responsibility","accountability", "reliability", "dependability", "consistency", "stability", "security", "safety",
    ],

    "frustration": 
    ["frustrated", "annoyed", "irritated", "upset", "discouraged", "fed up","frustrated", "annoyed", "irritated", "fed up", "disappointed", "exasperated", "upset", "stuck", "blocked",
     "muscle cramps", "internal panic", "drenched in sweat", "no appetite", "butterflies in stomach"
        ],

    "anger": 
    ["angry", "furious", "rage", "mad", "irritated", "hostile","angry", "furious", "rage", "mad", "irritated", "outraged", "annoyed", "fuming", "resentful", "agitated","enraged", "livid", "hostile", "bitter", "wrathful", "indignant", "cross", "infuriated", "seething","provoked",  "unforgiving", "unpleasantly", "unruly", "unsettled", "untrusting", "untrustworthy",
        ],

    "sarcasm": 
    ["yeah right", "totally", "sure", "as if", "good luck with that","yeah, right", "sure, whatever", "oh, really?", "that’s just great", "totally", "as if", "seriously?", "sure thing", 
        "that’s such a brilliant idea", "that’s what I was thinking", "well, aren’t you just clever?"
        ],

    "positive": positive_words,
    "negative": negative_words,
}

# --- Sentiment & Emotion Analysis ---
# Sentiment & Emotion Analysis
def analyze_sentiment_and_emotion(text: str) -> dict:
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    emotion_counts = defaultdict(int)

    for emo_label, word_list in emotion.items():
        for word in word_list:
            if ' ' in word and word in text_lower:
                emotion_counts[emo_label] += 1
            elif word in words:
                emotion_counts[emo_label] += 1

    pos = emotion_counts.pop("positive", 0)
    neg = emotion_counts.pop("negative", 0)

    if pos > neg:
        sentiment = "Positive"
    elif neg > pos:
        sentiment = "Negative"
    elif pos == neg and pos > 0:
        sentiment = "Mixed"
    else:
        sentiment = "Neutral"

    emotion_label = max(emotion_counts.items(), key=lambda x: x[1], default=("Neutral", 0))[0]

    return {
        "sentiment": sentiment,
        "emotion": emotion_label,
        "counts": dict(emotion_counts)
    }

def fast_generate_poetic_response(text: str) -> str:
    sentiment = analyze_sentiment_and_emotion(text)["sentiment"]
    return f"In a {sentiment.lower()} tone, here’s a poetic take:\n\n“{text}” 🌟"



def gemini_sentiment_tool(input_text: str) -> str:
    """Analyzes sentiment and emotion using Gemini 1.5 Flash."""
    prompt = f"""
    Analyze the sentiment and emotion of the following text:

    "{input_text}"
    Return a short and clear result like:
    Sentiment: Positive/Negative/Neutral
    Emotion: Joy/Anger/Sadness/etc.
    """
    response = model.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# Tool decorator for LangChain
@Tool(
    func=gemini_sentiment_tool,  # The function to be used
    name="gemini_sentiment_tool",  # Name of the tool
    description="Analyzes sentiment and emotion of the given text using the Gemini 1.5 Flash model."
)
def decorated_gemini_sentiment_tool(input_text: str) -> str:
    return gemini_sentiment_tool(input_text)

# Initialize agent with the tool
tools = [decorated_gemini_sentiment_tool]
agent = initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)














and my app.py code for Text Analysis & Sentiment Response   feature :                                                       from sentiment import agent, fast_generate_poetic_response,
from sentiment import agent, fast_generate_poetic_response, analyze_sentiment_and_emotion    
         # --- Sentiment Analysis & Creative Response --- 
elif page == "Text Analysis & Sentiment Response":
    st.title("📝 Text Analysis and Creative Sentiment Response Generator")
    st.markdown("### Enter your text below:")

    user_input = st.text_area("Input Text:", height=150)
    fast_mode = st.checkbox("⚡ Fast Mode (no LLM calls)", value=True)

    col1, col2 = st.columns(2)
    analyze_btn = col1.button("Analyze Sentiment")
    creative_btn = col2.button("Generate Creative Response")

    if user_input:
        if analyze_btn:
            sentiment_and_emotion = analyze_sentiment_and_emotion(user_input)
            st.subheader("📊 Analysis Result")
            st.markdown(f"**Action:** AnalyzeSentiment")
            st.markdown(f"**Input:** {user_input}")
            st.markdown(f"**Observation:** {sentiment_and_emotion}")

        if creative_btn:
            st.subheader("🎨 Creative Response")
            st.markdown(f"**Action:** GenerateCreativeResponse")
            st.markdown(f"**Input:** {user_input}")

            if fast_mode:
                poetic_response = fast_generate_poetic_response(user_input)
                st.markdown(f"**✨ Poetic Output:**\n\n{poetic_response}")
            else:
                with st.spinner("⏳ Generating response..."):
                    agent_response = agent.run(user_input)  # agent must be defined globally
                    st.markdown(f"**Agent Raw Response:**\n{agent_response}")
                    st.subheader("✨ Poetic Output")
                    st.markdown(agent_response)  # Display the agent's creative response

    else:
        st.info("Please enter some text above to begin.")

    st.markdown("---") analyze_sentiment_and_emotion       