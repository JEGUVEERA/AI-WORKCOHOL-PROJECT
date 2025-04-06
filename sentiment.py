import os
import json
import streamlit as st
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load .env file if it exists (for local development)
load_dotenv()

# Check if running on Streamlit Cloud (st.secrets is available)
if "GOOGLE_CREDENTIALS" in st.secrets:
    # Load credentials from secrets.toml
    service_account_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
else:
    # Load from local file path using .env variable
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

    if not credentials_path:
        raise ValueError("GOOGLE_CREDENTIALS_PATH environment variable is not set.")

    with open(credentials_path) as f:
        service_account_info = json.load(f)

# Now create credentials object
credentials = service_account.Credentials.from_service_account_info(service_account_info)



# Sentiment analysis function
def analyze_sentiment(text: str) -> str:
    positive_words = [
    "happy", "joy", "love", "great", "amazing", "wonderful", "brilliant", "cheerful", "delightful", "ecstatic",
    "fantastic", "grateful", "harmonious", "inspiring", "jubilant", "kind", "lively", "marvelous", "optimistic",
    "peaceful", "radiant", "spectacular", "thriving", "uplifting", "victorious", "warmhearted", "zealous",
    "accomplished", "admirable", "affectionate", "authentic", "benevolent", "blessed", "bountiful", "buoyant",
    "calm", "charming", "compassionate", "confident", "courageous", "courteous", "dazzling", "dedicated",
    "determined", "dynamic", "eager", "effervescent", "elevated", "empowered", "enchanting", "energetic",
    "enthusiastic", "exceptional", "exuberant", "faithful", "flourishing", "forgiving", "friendly", "generous",
    "gentle", "genuine", "glorious", "graceful", "grounded", "hardworking", "helpful", "honest", "hopeful",
    "humble", "illustrious", "impressive", "independent", "ingenious", "innovative", "intelligent",
    "invigorating", "joyous", "jovial", "keen", "legendary", "limitless", "lovable", "magnificent",
    "mindful", "motivated", "noble", "nurturing", "open-minded", "outstanding", "passionate", "patient",
    "peace-loving", "persevering", "persistent", "philanthropic", "playful", "positive", "powerful",
    "proactive", "productive", "prosperous", "proud", "radiant", "refreshing", "reliable", "resilient",
    "resourceful", "respectful", "responsible", "rewarding", "satisfied", "selfless", "sensible",
    "sincere", "skillful", "soothing", "spirited", "spontaneous", "strong", "successful", "supportive",
    "sympathetic", "thoughtful", "tolerant", "trustworthy", "unwavering", "valiant", "vibrant", "wise",
    "witty", "youthful", "zesty", "adventurous", "affluent", "amiable", "artistic", "aspiring",
    "authentic", "balanced", "breathtaking", "bright", "captivating", "carefree", "celebratory",
    "chivalrous", "classic", "colorful", "compelling", "congenial", "content", "creative",
    "cultivated", "daring", "decisive", "dedicated", "delicate", "diligent", "distinguished",
    "divine", "effortless", "elegant", "elated", "eloquent", "empathic", "empowered",
    "enlightened", "enterprising", "expressive", "exquisite", "fascinating", "fearless",
    "fertile", "festive", "flawless", "fortunate", "free-spirited", "fun-loving", "generative",
    "genius", "glamorous", "glowing", "graceful", "groundbreaking", "handsome", "healing",
    "heartwarming", "heroic", "high-spirited", "hopeful", "hospitable", "humorous", "idealistic",
    "imaginative", "immaculate", "industrious", "influential", "insightful", "intuitive",
    "inventive", "jolly", "jubilant", "keen", "laudable", "lively", "loving", "loyal",
    "magical", "majestic", "masterful", "meditative", "mesmerizing", "meticulous",
    "mind-blowing", "miraculous", "motivational", "natural", "neat", "nurturing",
    "observant", "omniscient", "opulent", "orderly", "original", "outgoing",
    "outstanding", "passionate", "peaceful", "perceptive", "perseverant",
    "persistent", "philosophical", "playful", "poetic", "polished", "popular",
    "practical", "precious", "priceless", "profound", "progressive", "pure",
    "purposeful", "quick-witted", "radiant", "reassuring", "refined", "refreshing",
    "rejuvenating", "remarkable", "resilient", "resourceful", "respectable",
    "revered", "rewarding", "romantic", "sagacious", "sensational", "sensuous",
    "serene", "sharp", "shining", "skillful", "smart", "sociable", "soulful",
    "sparkling", "spectacular", "spontaneous", "steadfast", "stunning", "suave",
    "sublime", "successful", "sufficient", "superb", "supportive", "sweet",
    "sympathetic", "talented", "tenacious", "tender", "thrilled", "tidy",
     "transformative", "trustworthy", "truthful", "unconditional",
    "unfailing", "unique", "uplifted", "valiant", "versatile", "vibrant",
    "visionary", "vivacious", "warm", "welcoming", "wise", "witty", "wonderful",
    "worthy", "youthful", "zealous", "zesty","good"
]

    negative_words = [
    "sad", "angry", "hate", "bad", "awful", "terrible", "horrible", "miserable", "depressed", "annoyed",
    "frustrated", "disappointed", "upset", "resentful", "unhappy", "gloomy", "hopeless", "pessimistic",
    "anxious", "worried", "stressed", "fearful", "nervous", "jealous", "insecure", "guilty", "ashamed",
    "regretful", "lonely", "isolated", "betrayed", "rejected", "hurt", "humiliated", "embarrassed",
    "offended", "defensive", "irritated", "hostile", "vengeful", "rude", "arrogant", "selfish",
    "greedy", "manipulative", "deceitful", "insincere", "dishonest", "corrupt", "cruel", "cold",
    "insensitive", "callous", "apathetic", "neglectful", "inconsiderate", "ungrateful", "lazy",
    "incompetent", "careless", "reckless", "clumsy", "useless", "worthless", "pathetic",
    "pointless", "meaningless", "hopeless", "tragic", "painful", "brutal", "savage", "sinister",
    "evil", "wicked", "malicious", "vindictive", "spiteful", "destructive", "dangerous",
    "toxic", "poisonous", "contaminated", "dirty", "filthy", "polluted", "nasty", "disgusting",
    "repulsive", "rotten", "vile", "horrendous", "shameful", "unforgiving", "harsh", "unfair",
    "displeasing", "dismal", "insulting", "distasteful", "disastrous", "frightening", "dangerous",
    "painful", "grieving", "sorrowful", "unfortunate", "tragic", "mournful", "unpleasant",
    "toxic", "disorienting", "blameful", "condemning", "unjust", "mean", "difficult", "untrustworthy",
    "divisive", "angst", "struggling", "bitter", "suspicious", "hostile", "dark", "oppressive", "disturbing",
    "hateful", "alienated", "horrible", "apathetic", "ugly", "irritating", "disappointing", "low", 
    "mean-spirited", "untrustworthy", "horrific", "devastating", "vicious", "ugly", "dreadful", "abominable", 
    "unbearable", "unfortunate", "scary", "undesirable", "unwelcome", "unnecessary", "desolate", 
    "sickening", "appalling", "unreliable", "hateful", "aggressive", "tormenting", "abusive", 
    "discomforting", "dismaying", "untrusting", "paranoid", "disgusted", "haggard", "unworthy", 
    "sour", "suffocating", "discontent", "doubtful", "unmotivated", "neglected", "paralyzing", 
    "harrowing", "unjustified", "unsatisfying", "despondent", "revolting", "pitiful", "unhappy", 
    "disillusioned", "defeatist", "distressing", "hopelessness", "grievous", "apathetic", "dreary", 
    "frustrating", "dreadful", "complicated", "undeserving", "helpless", "downhearted", "suffocating", 
    "mournful", "unfavorable", "aggravating", "sickly", "damning", "reprehensible", "off-putting", 
    "counterproductive", "self-destructive", "unsympathetic", "uncooperative", "toxic", "unpredictable", 
    "unsuccessful", "opposing", "debilitating", "unattainable", "miserably", "hurtful", "demoralizing", 
    "distasteful", "ungracious", "unreceptive", "persecutory", "sabotaging", "irking", "paranoiac", 
    "pathetically", "disillusioned", "uninspiring", "unfitting", "unimpressive", "unhealthy", 
    "negative", "irritating", "broken", "regret", "unfulfilled", "degraded", "contradictory", 
    "depressing", "disconnected", "disheartening", "inferior", "intolerable", "vulgar", "morose", 
    "insufficient", "unfortunate", "oppressive", "hollow", "detrimental", "harsh", "frightening", 
    "grueling", "unwilling", "reprehensible", "unrelenting", "disturbing", "inflexible", "ruinous", 
    "deficient", "failing", "unethical", "unfulfilling", "hostile", "unjust", "destructive", 
    "disruptive", "worthless", "rejected", "downhearted", "resentful", "lacking", "resenting", 
    "oppositional", "obnoxious", "unappealing", "overbearing", "unforgiving", "pointless", 
    "insulting", "tragic", "imperfect", "wretched", "worrisome", "unfit", "discouraging", "dark", 
    "morbid", "regrettable", "rejected", "dismaying", "undesirable", "heartbreaking", "unsavory", 
    "undermining", "dejected", "despairing", "horrifying", "dread", "opposing", "unwanted", "unfocused", 
    "shocking", "grating", "unsuccessful", "compromised", "unworthy", "unpleasant", "terrifying", 
    "scornful", "intolerant", "ugly", "uncompromising", "disturbing", "discouraged", "exasperated", 
    "troublesome", "demotivating", "unapproachable", "unreliable", "distressed", "divisive", 
    "inconsiderate", "unwanted", "unsatisfactory", "destructive", "grievous", "hopeless", "shameful", 
    "pointless", "useless", "intolerable", "upsetting", "resenting", "repulsive", "bitter", "devastating",
    "discouraging", "unforgiving", "downcast", "unsuccessful", "ruining", "toxic", "draining", "stifling", 
    "conflicting", "distasteful", "unproductive", "blaming", "unsuitable", "tragically", "unfriendly",
    "infuriating", "agonizing", "unrecoverable", "unethical", "paralyzing", "unsolicited", "horrendous",
    "unsound", "unhelpful", "unresolvable", "failing", "unresolved", "lousy", "regretful"
]    
    text = text.lower()
    positive_count = sum(word in text for word in positive_words)
    negative_count = sum(word in text for word in negative_words)

    if positive_count > negative_count:
        return "✅ The sentiment of the text is **positive**."
    elif negative_count > positive_count:
        return "⚠️ The sentiment of the text is **negative**."
    else:
        return "ℹ️ The sentiment of the text is **neutral**."

def generate_creative_response(text: str) -> str:
    return f"✨ *Creative Response*: Imagine a world where \"{text}\" becomes the heart of a magical story. What adventures would unfold?"

# Define LangChain tools

tools = [
    Tool(
        name="AnalyzeSentiment",
        func=analyze_sentiment,
        description="Analyze sentiment of a text (positive, negative, neutral)"
    ),
    Tool(
        name="GenerateCreativeResponse",
        func=generate_creative_response,
        description="Generate a creative and engaging response based on the input text"
    ),
]

# Memory and prompt
memory = ConversationBufferMemory(memory_key="chat_history")
prompt_template = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
You are an advanced text analysis and creative response agent. You have access to the following tools:
- AnalyzeSentiment: Analyzes sentiment of the text.
- GenerateCreativeResponse: Generates a creative and engaging response based on the input text.
Chat History: {chat_history}
User Input: {input}
"""
)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)


