import re
from collections import defaultdict

positive_words = ["amazing", "wonderful", ...]
negative_words = ["infuriating", "unethical", ...]
emotion_keywords = {
    "happiness": ["happy", "joy", "excited"],
    "anger": ["angry", "furious"],
    "sadness": ["sad", "melancholy"],
    "fear": ["afraid", "terrified"],
    "neutral": ["neutral"]
}

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
    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if re.search(rf"\b{re.escape(word)}\b", text_lower):
                emotion_counts[emotion] += 1

    sentiment = (
        "positive" if sentiment_score > 0 else
        "negative" if sentiment_score < 0 else
        "neutral"
    )
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1], default=("neutral", 0))[0]

    return {
        "sentiment": sentiment,
        "dominant_emotion": dominant_emotion,
        "emotion_scores": dict(emotion_counts)
    }

def generate_creative_response(text: str) -> str:
    return f"✨ Here's a creative twist: {text[::-1]}"
