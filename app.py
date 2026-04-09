from fastapi import FastAPI
import pickle
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

model = pickle.load(open('department_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_sentiment(text):
    text_lower = text.lower()
    
    # Force negative for complaint keywords
    negative_words = ["not", "no", "issue", "problem", "error", "failed", "unable"]
    
    if any(word in text_lower for word in negative_words):
        return "Negative"
    
    score = analyzer.polarity_scores(text)['compound']
    
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_priority(text, sentiment):
    text = text.lower()
    
    if "urgent" in text:
        return "Critical"
    
    if sentiment == "Negative":
        return "High"
    elif sentiment == "Positive":
        return "Low"
    else:
        return "Medium"

@app.get("/predict")
def predict(text: str):
    text_clean = clean_text(text)
    text_tfidf = tfidf.transform([text_clean])
    
    pred = model.predict(text_tfidf)
    department = le.inverse_transform(pred)[0]
    
    sentiment = get_sentiment(text)
    priority = get_priority(text, sentiment)

    return {
        "department": department,
        "sentiment": sentiment,
        "priority": priority
    }