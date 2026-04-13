# AI-Driven Citizen Grievance & Sentiment Analysis System

## 📌 Project Overview

This project is an AI-based system that analyzes citizen complaints and automatically:

* Predicts the relevant **department**
* Identifies **sentiment** (Positive / Negative / Neutral)
* Assigns **priority level** (Low / Medium / High / Critical)

The goal is to reduce manual effort and improve complaint resolution efficiency.

---

## 🚀 Features

* Text preprocessing (cleaning, stopwords removal, lemmatization)
* Department classification using Machine Learning
* Sentiment analysis using rule-based + VADER
* Priority detection based on complaint severity
* FastAPI-based REST API for real-time predictions

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* NLTK
* FastAPI
* TF-IDF Vectorization

---

## 📊 Model Workflow

1. Input complaint text
2. Text preprocessing (cleaning + lemmatization)
3. TF-IDF vectorization
4. Department prediction using trained model
5. Sentiment detection
6. Priority assignment

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn nltk scikit-learn vaderSentiment
```

### 2. Run API

```bash
uvicorn app:app --reload
```

### 3. Open in browser

```
http://127.0.0.1:8000/docs
```

---

## 🔍 Example Input

```
My electricity bill is wrong and I was overcharged
```

## ✅ Output

```
Department: Electricity
Sentiment: Negative
Priority: High
```

---

## 📁 Project Structure

```
├── app.py
├── DS_ML_Sentiment.ipynb
├── README.md
```

---

## 📌 Conclusion

This project demonstrates how AI and NLP can be used to automate grievance handling and improve service efficiency.

---
