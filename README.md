# AI-Driven Citizen Grievance & Sentiment Analysis System
**Infotact Technical Internship Program — Project 1: Government & Public Sector**

An end-to-end NLP pipeline that automatically classifies citizen complaints into government departments, analyses sentiment using a trained ML model, and assigns urgency priority — all served via a FastAPI REST API.

---

## Problem Statement

Citizens filing grievances often face manual routing delays and no urgency triage. This system automates that process: a raw complaint is submitted, and the AI instantly returns the correct department, emotional tone, and priority level.

---

## Project Structure

```
├── DS_ML_Sentiment_Complete.ipynb   # Full ML pipeline (training + evaluation)
├── app.py                           # FastAPI inference server
├── department_model.pkl             # Trained department classifier
├── tfidf_vectorizer.pkl             # TF-IDF vectorizer (department)
├── label_encoder.pkl                # Label encoder (department)
├── sentiment_model.pkl              # Trained sentiment classifier (Week 3)
├── tfidf_sentiment.pkl              # TF-IDF vectorizer (sentiment)
├── le_sentiment.pkl                 # Label encoder (sentiment)
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| NLP & ML | Scikit-learn, NLTK, vaderSentiment |
| Vectorization | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Models | Logistic Regression (department + sentiment) |
| API Serving | FastAPI + Uvicorn |
| EDA & Visualization | Matplotlib, Seaborn, WordCloud |
| Serialization | Pickle |

---

## Four-Week Roadmap

### Week 1 — Data Collection, Text Cleaning & EDA
- Loaded `complaints_processed.csv` (CFPB consumer complaints dataset)
- Text preprocessing pipeline: lowercasing, URL removal, special character stripping, stopword removal, lemmatization
- Generated Word Cloud of most frequent complaint terms
- Plotted N-gram (bigram) frequency distribution
- Visualised department class distribution

### Week 2 — Department Categorization
- Converted cleaned text to numerical vectors using **TF-IDF** (`max_features=5000`)
- Trained a **Logistic Regression** classifier to predict one of 5 departments:
  - `credit_card`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking`
- Applied **5-fold Stratified Cross-Validation** → Mean Accuracy: **87.16%**

### Week 3 — Sentiment Analysis (Trained ML Model)
- Auto-labeled the dataset using VADER to produce `Positive / Neutral / Negative` ground-truth labels
- Trained a dedicated **Logistic Regression sentiment classifier** on a separate TF-IDF vectorizer (`ngram_range=(1,2)`, `class_weight='balanced'`)
- Evaluated using **Macro F1-score** (project KPI) and Confusion Matrix
- Implemented keyword-enhanced **Priority/Urgency Scoring**: `Critical → High → Medium → Low`

### Week 4 — Evaluation, Serialization & API Deployment
- Generated Confusion Matrix and Classification Report for the department model
- Serialized all 6 model artefacts to `.pkl` files
- Wrapped the pipeline in a **FastAPI** application
- Tested end-to-end inference with multiple complaint samples

---

## Model Performance

### Department Classifier
| Metric | Score |
|---|---|
| Cross-Val Mean Accuracy | 87.16% |
| Cross-Val Std | ±0.10% |

| Department | Precision | Recall | F1 |
|---|---|---|---|
| credit_card | 0.79 | 0.78 | 0.79 |
| credit_reporting | 0.90 | 0.93 | 0.92 |
| debt_collection | 0.81 | 0.72 | 0.76 |
| mortgages_and_loans | 0.85 | 0.83 | 0.84 |
| retail_banking | 0.86 | 0.87 | 0.86 |

### Sentiment Classifier
- Evaluated using **Macro F1-score** to ensure fair scoring across minority classes (Neutral)
- Classes: `Negative`, `Neutral`, `Positive`

---

## Setup & Installation

```bash
# Install dependencies
pip install pandas scikit-learn vaderSentiment fastapi uvicorn wordcloud matplotlib seaborn nltk

# Run the notebook end-to-end first to generate all .pkl files
# Then start the API server
python -m uvicorn app:app --reload
```

---

## API Usage

**Endpoint:** `GET /predict?text=<your complaint here>`

**Example Request:**
```
GET http://127.0.0.1:8000/predict?text=My credit card was charged twice and nobody is responding
```

**Example Response:**
```json
{
  "Department": "credit_card",
  "Sentiment": "Negative",
  "Priority": "High"
}
```

**Priority Levels:**
- `Critical` — fraud, legal threats, hacking, life/safety, escalation keywords
- `High` — payment failures, account blocks, long wait times, negative sentiment
- `Medium` — neutral tone, no urgent keywords
- `Low` — positive feedback

**Interactive Docs:** `http://127.0.0.1:8000/docs`

---

## Dataset

Consumer Financial Protection Bureau (CFPB) complaints dataset (`complaints_processed.csv`).
Contains real-world citizen financial complaints mapped to product/department categories.

> Note: The raw dataset is excluded from the repository per `.gitignore` data security guidelines.

---

## Author

Developed as part of the **Infotact Technical Internship Program**
Track: Data Science & Machine Learning | Domain: Government & Public Sector
