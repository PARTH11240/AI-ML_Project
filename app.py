from fastapi import FastAPI
import pickle
import re

app = FastAPI()

# ── Department classifier artefacts ──────────────────────────────────────────
model = pickle.load(open('department_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
le    = pickle.load(open('label_encoder.pkl',    'rb'))

# ── Sentiment classifier artefacts (trained ML model) ────────────────────────
sentiment_model  = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf_sentiment  = pickle.load(open('tfidf_sentiment.pkl', 'rb'))
le_sentiment     = pickle.load(open('le_sentiment.pkl',    'rb'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def get_sentiment(text):
    """Uses the trained Logistic Regression sentiment model."""
    cleaned = clean_text(text)
    vec     = tfidf_sentiment.transform([cleaned])
    return le_sentiment.inverse_transform(sentiment_model.predict(vec))[0]


def get_priority(text, sentiment):
    text_lower = text.lower()

    critical_words = [
        "urgent", "urgently", "urgency", "immediately", "immediate",
        "asap", "right now", "as soon as possible", "emergency", "critical",
        "stolen", "theft", "robbery", "fraud", "fraudulent", "scam", "cheated",
        "hacked", "hack", "hacking", "cyber attack", "phishing",
        "unauthorized transaction", "unauthorized access", "identity theft",
        "data breach", "leaked data", "police", "fir", "legal action",
        "court", "lawsuit", "sue", "lawyer", "consumer court", "rbi complaint",
        "banking ombudsman", "legal notice", "life", "death", "dying",
        "hospital", "medical", "accident", "danger", "dangerous", "threat",
        "all money gone", "entire savings", "life savings", "account emptied",
        "already complained", "complaint not resolved", "second complaint",
        "third complaint", "escalate", "escalating", "escalation",
        "higher authority", "senior manager", "ceo", "nodal officer",
        "deadline", "due date", "expiry", "expires today",
    ]

    high_words = [
        "not working", "not functioning", "stopped working",
        "transaction failed", "payment failed", "transfer failed",
        "money deducted", "amount deducted", "balance deducted",
        "refund pending", "refund not received", "waiting for refund",
        "account blocked", "account locked", "card blocked",
        "card not working", "atm not working", "otp not received",
        "emi bounced", "cheque bounced", "overcharged", "double charged",
        "wrong amount", "not received", "never received", "still not received",
        "no response", "no reply", "nobody helping",
        "days", "weeks", "months", "long time",
    ]

    for word in critical_words:
        if word in text_lower:
            return "Critical"

    for word in high_words:
        if word in text_lower:
            return "High"

    if sentiment == "Negative":
        return "High"
    elif sentiment == "Positive":
        return "Low"
    return "Medium"


@app.get("/predict")
def predict(text: str):
    clean      = clean_text(text)
    vec        = tfidf.transform([clean])
    department = le.inverse_transform(model.predict(vec))[0]
    sentiment  = get_sentiment(text)
    priority   = get_priority(text, sentiment)

    return {
        "Department" : department,
        "Sentiment"  : sentiment,
        "Priority"   : priority
    }
