# 📈 AI-Powered Financial Decision Support System

An end-to-end machine learning project that combines market data, technical indicators, and financial news sentiment to generate intelligent **Buy**, **Sell**, or **Hold** recommendations through a custom **Signal Logic Model (SLM)**.

---

## 🚀 Overview

Financial markets are influenced by multiple factors such as price trends, volatility, trading volume, technical indicators, and news sentiment. Rather than relying on a single machine learning model, this project combines multiple specialized models to support more informed financial decision-making.

The system processes historical market data and financial news, extracts meaningful features, generates predictions using multiple ML models, and combines them through a custom **Signal Logic Model (SLM)** to produce an actionable trading recommendation.

---

## 🏗️ System Architecture

> **Architecture Diagram**

<p align="center">
  <img src="architecture.png" width="900">
</p>

---

## ✨ Features

- Historical Market Data Collection
- Financial News Collection
- Feature Engineering using Technical Indicators
- Trend Prediction
- Market Regime Classification
- Volatility Prediction
- News Sentiment Analysis
- Signal Logic Model (SLM)
- Buy / Sell / Hold Recommendation Engine

---

## 📊 Data Sources

### Market Data
- Yahoo Finance
- Alpha Vantage

### News Data
- NewsAPI
- Finnhub
- Yahoo Finance News
- Google News RSS

---

## 📈 Feature Engineering

The following technical indicators are generated before model training:

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Exponential Moving Averages (EMA 20, 50, 200)
- Average Directional Index (ADX)
- Momentum
- Returns
- Volume Ratio
- Breakout Indicators

These engineered features are used as inputs for the machine learning models.

---

## 🤖 Machine Learning Models

| Model | Purpose |
|--------|---------|
| Trend Model | Predicts short-term market direction |
| Regime Classifier | Detects Bull, Bear, or Sideways markets |
| Volatility Model | Estimates future market volatility |
| News Sentiment Model | Measures the impact of financial news |

---

## 🧠 Signal Logic Model (SLM)

The Signal Logic Model combines outputs from all trained models using:

- Signal Weighting
- Market Regime Adjustment
- Risk Evaluation
- Decision Logic

The final output is an actionable recommendation:

- 🟢 Buy
- 🟡 Hold
- 🔴 Sell

---

## 🛠️ Tech Stack

**Programming Language**

- Python

**Libraries**

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

**APIs**

- Yahoo Finance
- Alpha Vantage
- NewsAPI
- Finnhub

---

## 📂 Project Structure

```
Financial-Decision-Support-System
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── src/
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── trend_model.py
│   ├── regime_classifier.py
│   ├── volatility_model.py
│   ├── sentiment_model.py
│   └── slm_engine.py
│
├── outputs/
│
├── architecture.png
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Financial-Decision-Support-System.git
```

Move into the project directory

```bash
cd Financial-Decision-Support-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

## 📌 Future Improvements

- Deep Learning-based forecasting
- Transformer-based sentiment analysis
- Real-time market monitoring
- Portfolio optimization
- Interactive Streamlit dashboard
- Reinforcement Learning-based trading strategies

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It should not be considered financial or investment advice. Always conduct your own research before making investment decisions.

---

## 👨‍💻 Author

**Abhishek Raghav**

MBA Business Analytics | Data Analytics & Machine Learning Enthusiast

- LinkedIn: https://www.linkedin.com/in/your-profile
- GitHub: https://github.com/your-username

---

### ⭐ If you found this project interesting, consider giving it a star!
