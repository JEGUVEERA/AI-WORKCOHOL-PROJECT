from transformers import pipeline

analyzer = pipeline("sentiment-analysis")

def analyze_sentiment_and_respond(text):
    result = analyzer(text)[0]
    label, score = result["label"], result["score"]
    if label == "POSITIVE":
        return f"😊 That sounds positive! (Confidence: {score:.2f})"
    elif label == "NEGATIVE":
        return f"😞 That seems a bit negative. (Confidence: {score:.2f})"
    else:
        return f"😐 That’s neutral. (Confidence: {score:.2f})"
