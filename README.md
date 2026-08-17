# IntelliTradeX

### AI-Powered Stock Prediction & Sentiment Analysis Platform

**Smart India Hackathon Problem Statement:** SIH25127

IntelliTradeX is an AI-powered financial analytics platform that combines **Machine Learning, Deep Learning, and NLP** to analyze stock market trends, predict potential price movements, and understand investor sentiment from financial news.

The platform transforms complex market data into **interactive visual insights and AI-driven predictions**, helping users make more informed, data-driven decisions.

---

## 🚀 Key Features

* 📈 **Stock Price Prediction** using LSTM and Machine Learning models
* 📰 **Financial News Sentiment Analysis** using NLP
* 📊 **Interactive Stock Market Dashboard**
* 🔮 **AI-Based Market Trend Insights**
* 📉 **Historical Price & Prediction Visualization**
* 🧠 **Hybrid ML Approach** combining LSTM and Random Forest
* ⚡ **Automated Stock Data Fetching**
* 💾 **Prediction & Analysis Data Storage**

---

## 🧩 How It Works

```text
Stock Market Data ──┐
                    ├──> Data Preprocessing ──> ML Model ──> Price Prediction
Financial News ─────┘
                         │
                         └──> NLP Sentiment Analysis
                                      │
                                      ▼
                           Interactive Dashboard
```

The system follows a modular pipeline:

1. **Data Collection** – Fetches historical and market-related stock data.
2. **Preprocessing** – Cleans, transforms, and prepares data for analysis.
3. **Prediction** – Uses LSTM/ML models to identify potential stock price trends.
4. **Sentiment Analysis** – Analyzes financial news to determine market sentiment.
5. **Visualization** – Presents predictions, trends, and sentiment insights through an interactive dashboard.

---

## 🛠️ Tech Stack

### Frontend & Dashboard

* **Streamlit** – Interactive financial dashboard
* **Plotly / Charts** – Market data visualization
* **Python** – Application development

### Machine Learning & AI

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Joblib**

### Models

* **LSTM** – Time-series stock price prediction
* **Random Forest** – Machine learning-based prediction
* **NLP Sentiment Analysis** – Financial news sentiment classification

### Data & APIs

* **Yahoo Finance / Financial APIs**
* Historical and real-time market data

### Development Tools

* **VS Code**
* **Git & GitHub**
* **Jupyter Notebook**

---

## 📂 Project Structure

```text
IntelliTradeX/
│
├── enhanced_features.py       # Additional AI & dashboard features
├── fetch_stock_data.py        # Fetches stock market data
├── preprocess_store_data.py   # Data preprocessing and storage
├── sentiment_analysis.py      # Financial news sentiment analysis
├── train_lstm_model.py        # LSTM model training
├── run_model.py               # Runs prediction pipeline
├── stock_dashboard.py         # Streamlit dashboard
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🧠 AI & ML Pipeline

**Market Data → Preprocessing → Feature Engineering → LSTM / ML Model → Prediction**

**Financial News → NLP Processing → Sentiment Classification → Sentiment Score**

The combination of **quantitative market prediction** and **qualitative sentiment analysis** provides a more comprehensive view of market behavior.

---

## 🎯 Objective

IntelliTradeX aims to make financial market analysis more **accessible, intelligent, and data-driven** by bringing stock prediction and sentiment analysis together in a single platform.

> **Predict the trend. Understand the sentiment. Make smarter decisions.**

---

## ⚠️ Disclaimer

IntelliTradeX is an educational and analytical project. Its predictions and insights are **not financial advice** and should not be used as the sole basis for investment decisions.
