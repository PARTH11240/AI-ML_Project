from fastapi import FastAPI
import pickle
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

model = pickle.load(open('department_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
le    = pickle.load(open('label_encoder.pkl',    'rb'))

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_sentiment(text):
    text_lower = text.lower()

    negative_words = [
        # Problem words
        "not", "no", "issue", "issues", "problem", "problems", "error",
        "errors", "failed", "failure", "unable", "broken", "bug",
        # Financial fraud
        "stolen", "theft", "fraud", "scam", "cheat", "cheated", "cheating",
        "fake", "false", "forged", "unauthorized", "illegal", "crime",
        "hacked", "hack", "breach", "leaked", "exposed",
        # Complaint words
        "complaint", "complain", "complaining", "dispute", "disputing",
        "wrong", "incorrect", "mistake", "mistakes", "error", "errors",
        # Negative feelings
        "bad", "terrible", "horrible", "worst", "useless", "pathetic",
        "disgusting", "unacceptable", "ridiculous", "unfair", "awful",
        "dreadful", "appalling", "shocking", "outrageous", "nonsense",
        "frustrated", "frustrating", "frustration", "angry", "anger",
        "annoyed", "annoying", "upset", "furious", "disappointed",
        "disappointing", "disappointment", "fed up", "sick", "tired",
        # Denial/rejection
        "denied", "deny", "rejected", "reject", "refused", "refuse",
        "blocked", "suspended", "banned", "closed", "terminated",
        "cancelled", "revoked", "deactivated", "disabled",
        # Missing/lost
        "lost", "missing", "disappeared", "vanished", "gone",
        "not received", "never received", "not working", "not functioning",
        "not responding", "no response", "no reply", "ignored",
        # Payment issues
        "charged", "overcharged", "double charged", "extra charge",
        "hidden charge", "unauthorized charge", "deducted", "deduction",
        "debited", "money gone", "money missing", "money lost",
        "refund not", "no refund", "pending refund",
        # Service issues
        "delay", "delayed", "late", "slow", "poor", "worst service",
        "bad service", "pathetic service", "no service", "no help",
        "helpless", "hopeless", "useless service", "terrible service",
        # Negations
        "cant", "cannot", "wont", "dont", "didnt", "doesnt",
        "havent", "hasnt", "hadnt", "shouldnt", "wouldnt", "couldnt",
        "never", "nothing", "nobody", "nowhere", "neither", "nor",
        # Account issues
        "account locked", "account blocked", "account suspended",
        "account closed", "access denied", "login failed",
        "password reset", "otp not", "verification failed",
        # Loan/mortgage issues
        "defaulted", "overdue", "penalty", "penalized", "interest",
        "emi failed", "loan rejected", "mortgage denied",
        # Others
        "harassment", "harassing", "threatening", "threatened",
        "blackmail", "extortion", "abuse", "abused", "mistreated",
        "negligence", "negligent", "irresponsible", "incompetent",
        "unresolved", "not resolved", "still pending", "no action",
        "waste", "wasted", "time waste", "money waste"
    ]

    positive_words = [
        # Thank you
        "thank", "thanks", "thankful", "grateful", "gratitude",
        "appreciate", "appreciated", "appreciation",
        # Good experience
        "good", "great", "excellent", "amazing", "wonderful",
        "fantastic", "outstanding", "superb", "brilliant", "perfect",
        "best", "awesome", "splendid", "magnificent", "exceptional",
        # Satisfied
        "happy", "happily", "happiness", "satisfied", "satisfaction",
        "pleased", "pleasing", "delighted", "content", "glad",
        # Resolved
        "resolved", "resolution", "fixed", "solved", "solution",
        "working", "works", "working fine", "working well",
        "processed", "completed", "done", "successful", "success",
        # Helpful
        "helpful", "helped", "help received", "assisted", "support",
        "supported", "cooperative", "professional", "efficient",
        "prompt", "quick", "fast", "smooth", "easy", "convenient",
        # Positive feedback
        "recommend", "recommended", "love", "loved", "like",
        "impressed", "impressive", "well done", "good job",
        "keep it up", "nice", "polite", "friendly", "kind", "caring"
    ]

    # Check negative first
    for word in negative_words:
        if word in text_lower:
            return "Negative"

    # Check positive
    for word in positive_words:
        if word in text_lower:
            return "Positive"

    # Fall back to VADER
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_priority(text, sentiment):
    text_lower = text.lower()

    critical_words = [
        # Urgency words
        "urgent", "urgently", "urgency", "immediately", "immediate",
        "asap", "right now", "as soon as possible", "right away",
        "emergency", "critical", "serious", "severe", "extreme",
        # Crime/fraud
        "stolen", "theft", "robbery", "robbed", "burglary",
        "fraud", "fraudulent", "scam", "scammer", "cheated",
        "hacked", "hack", "hacking", "cyber attack", "phishing",
        "unauthorized transaction", "unauthorized access",
        "identity theft", "data breach", "leaked data",
        # Legal threats
        "police", "police complaint", "fir", "legal action",
        "court", "lawsuit", "sue", "lawyer", "attorney",
        "consumer court", "rbi complaint", "banking ombudsman",
        "legal notice", "going to court",
        # Life/safety
        "life", "death", "dying", "hospital", "medical",
        "accident", "danger", "dangerous", "threat", "threatening",
        # Money urgency
        "all money gone", "entire savings", "life savings",
        "retirement money", "last money", "no money left",
        "account emptied", "zero balance", "wiped out",
        # Already escalated
        "already complained", "complaint not resolved",
        "second complaint", "third complaint", "escalate",
        "escalating", "escalation", "higher authority",
        "senior manager", "ceo", "nodal officer",
        # Time sensitive
        "deadline", "due date", "expiry", "expires today",
        "last date", "today only", "by tonight", "by morning"
    ]

    high_words = [
        "not working", "not functioning", "stopped working",
        "transaction failed", "payment failed", "transfer failed",
        "money deducted", "amount deducted", "balance deducted",
        "refund pending", "refund not received", "waiting for refund",
        "account blocked", "account locked", "card blocked",
        "card not working", "atm not working", "otp not received",
        "loan emi", "emi bounced", "cheque bounced",
        "overcharged", "double charged", "wrong amount",
        "not received", "never received", "still not received",
        "no response", "no reply", "nobody helping",
        "days", "weeks", "months", "long time", "since long"
    ]

    # Check Critical first
    for word in critical_words:
        if word in text_lower:
            return "Critical"

    # Check High
    for word in high_words:
        if word in text_lower:
            return "High"

    # Based on sentiment
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
