# from sentiment.py
import re
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

load_dotenv()

# Initialize Gemini LLM
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

## Simplified example positive/negative keywords
positive_words = [
    "happy", "joy", "love", "great", "amazing", "wonderful", "brilliant", "cheerful", "delightful", "ecstatic",
    "welcoming", "wise", "witty", "worthy", "youthful", "zealous", "zesty", "good","happy", "joy", "love", "great", "amazing", "wonderful", "brilliant", "cheerful", "delightful", "ecstatic",
]

negative_words = [
    "sad", "angry", "hate", "bad", "awful", "terrible", "horrible", "miserable", "depressed", "annoyed",
    "frustrated", "disappointed", "upset", "resentful", "unhappy", "gloomy", "hopeless", "anxious", "worried","sad", "angry", "hate", "bad", "awful", "terrible", "horrible", "miserable", "depressed", "annoyed",
    "frustrated", "disappointed", "upset", "resentful", "unhappy", 
]

# Emotion mapping
emotion = {
    "joy": 
    ["happy", "joy", "delighted", "cheerful", "smile", "grateful", "optimistic", "excited","happy", "joy", "delighted", "cheerful", "content", "smile", "grateful", "excited", "lively", "radiant","optimistic", "laugh", "gleeful", "blissful", "ecstatic", "jubilant", "elated", "satisfied", 
    ],

    "frustration": 
    ["frustrated", "annoyed", "irritated", "upset", "discouraged", "fed up","frustrated", "annoyed", "irritated", "fed up", "disappointed",  "irritably impatient"
    ],

    "confidence": 
    ["confident", "bold", "capable", "assured", "certain", "secure","confident", "assured", "secure", "positive", "certain", "self-assured",  "assertive"
        ],

    "sadness": 
    ["sad", "depressed", "unhappy", "heartbroken", "miserable", "gloomy","sad", "sorrow", "grief", "depressed", "unhappy", "heartbroken", 
     ],

    "anxiety": 
    ["nervous", "anxious", "worried", "uneasy", "restless", "panicked","nervous", "anxious", "worried", "uneasy", "tense", "scared", "panicked", 
        "frozen with fear", "muscle cramps", "internal panic", "drenched in sweat", "no appetite", "butterflies in stomach"
        ],

    "anger": 
    ["angry", "furious", "rage", "mad", "irritated", "hostile","angry", "furious", "rage", "mad", "irritated", "outraged", "annoyed", "fuming", 
        "unrelenting", "unforgiving", "unpleasantly", "unruly", "unsettled", "untrusting", "untrustworthy",
        ],

    "sarcasm": 
    ["yeah right", "totally", "sure", "as if", "good luck with that","yeah, right", "sure, whatever", "oh, really?", "that’s just great", 
        "how original", "that’s such a brilliant idea", "that’s what I was thinking", "well, aren’t you just clever?"
        ],

    "positive": positive_words,
    "negative": negative_words,
}

def analyze_sentiment_and_emotion(text: str) -> dict:
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    emotion_counts = defaultdict(int)

    for emo_label, word_list in emotion.items():
        for word in word_list:
            if ' ' in word:
                if word in text_lower:
                    emotion_counts[emo_label] += 1
            elif word in words:
                emotion_counts[emo_label] += 1

    # Sentiment Scoring
    pos, neg = emotion_counts.get("positive", 0), emotion_counts.get("negative", 0)
    if pos > neg:
        sentiment = "Positive"
    elif neg > pos:
        sentiment = "Negative"
    elif pos == neg and pos > 0:
        sentiment = "Mixed"
    else:
        sentiment = "Neutral"

    # Final cleanup
    emotion_counts.pop("positive", None)
    emotion_counts.pop("negative", None)

    # Most likely emotion
    emotion_label = max(emotion_counts.items(), key=lambda x: x[1], default=("Neutral", 0))[0]
    
    return {
        "sentiment": sentiment,
        "emotion": emotion_label,
        "counts": dict(emotion_counts)
    }
def generate_poetic_response(text: str) -> str:
    sentiment = analyze_sentiment_and_emotion(text)["sentiment"]
    prompt = f"The sentiment is {sentiment}. Create a poetic response to:\n{text}"
    return model.invoke([HumanMessage(content=prompt)]).content


tools = [
    Tool(
        name="SentimentEmotionTool",
        func=analyze_sentiment_and_emotion,
        description="Analyzes sentiment and emotion in a given text"
    ),
    Tool(
        name="PoeticResponseTool",
        func=generate_poetic_response,
        description="Generates poetic response based on sentiment"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# from app_main.py
from sentiment import agent, analyze_sentiment_and_emotion, generate_poetic_response


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
            st.markdown(f"Action: AnalyzeSentiment")
            st.markdown(f"Action Input: {user_input}")

            sentiment_and_emotion = analyze_sentiment_and_emotion(user_input)
            st.markdown(f"Observation: {sentiment_and_emotion}")
        else:
            st.warning("Please enter some text for sentiment analysis.")

    # Creative AI Response
    if col2.button("Generate Creative Response"):
        if user_input:
            agent_response = agent.run(user_input)
            creative_response = generate_poetic_response(user_input, sentiment_and_emotion["sentiment"])  # Sentiment is passed here
            # Display the response in structured format
            st.subheader("🎨 Creative Response")
            st.markdown(f"Action: GenerateCreativeResponse")
            st.markdown(f"Action Input: {user_input}")
            st.markdown(f"Generated Response: {agent_response}")
            st.subheader("✨ Creative Response")
            st.markdown(f"Generated Creative Response: {creative_response}")
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
            st.markdown("Thought: Do I need to use a tool? Yes")
            st.markdown("Action: AnalyzeSentiment")
            st.markdown(f"Action Input: {user_input}")
            st.markdown(f"Observation: {sentiment}")
            st.markdown("Thought: Do I need to use a tool? No")
            st.markdown("AI Final Response:")
            st.info(agent_output)

        # Sentiment Badge
        st.markdown("### 🏷 Sentiment Result")
        sentiment_color = "🟢 Positive" if "positive" in sentiment["sentiment"].lower() else "🔴 Negative" if "negative" in sentiment["sentiment"].lower() else "🟡 Neutral"
        st.success(f"Sentiment Analysis: *{sentiment_color}* — {sentiment['sentiment']}")

        ImportError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/ai-workcohol-project----ai-for-marketing/app_main.py", line 46, in <module>
    from sentiment import agent, analyze_sentiment_and_emotion, generate_poetic_response
