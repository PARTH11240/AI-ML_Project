# 🧠 AI-Based Grievance Classification & Prioritization System

## 📌 Project Overview

This project is an AI-powered Natural Language Processing (NLP) system that automatically analyzes customer complaints and performs:

* 🏷️ **Department Classification** (e.g., credit card, banking, reporting)
* 😊 **Sentiment Analysis** (Positive, Negative, Neutral)
* ⚡ **Priority Assignment** (Low, Medium, High, Critical)

The system helps organizations **automate complaint handling and prioritize urgent issues**, reducing manual effort and improving response time.

---

## 🚀 Features

* Text preprocessing and cleaning
* TF-IDF based feature extraction
* Logistic Regression classification model
* Sentiment analysis using VADER + rule-based enhancement
* Priority scoring system
* FastAPI deployment for real-time predictions

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* NLP (TF-IDF)
* VADER Sentiment Analysis
* FastAPI

---

## ⚙️ How It Works

1. User inputs complaint text
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical format
4. ML model predicts department
5. Sentiment analysis determines emotional tone
6. Priority is assigned based on sentiment and keywords
7. Output is returned via API

---

## 📂 Project Structure

```
app.py
department_model.pkl
tfidf_vectorizer.pkl
label_encoder.pkl
requirements.txt
```

---

## ▶️ How to Run Locally

1. Clone repository:

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run FastAPI server:

```
python -m uvicorn app:app --reload
```

4. Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 📊 Example Input & Output

### Input:

```
I am unable to use my credit card, very frustrating
```

### Output:

```json
{
  "department": "credit_card",
  "sentiment": "Negative",
  "priority": "High"
}
```

---

## 🎯 Business Impact

* Automates complaint routing
* Reduces manual workload
* Identifies urgent issues quickly
* Improves customer satisfaction

---

## 📌 Future Improvements

* Use advanced models like BERT
* Build a user-friendly web interface
* Add dashboard for analytics
* Improve sentiment accuracy

---

## 👨‍💻 Author

Parth Vadnagra
MBA - Business Intelligence

---
