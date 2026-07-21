import joblib
import pandas as pd
import json


trend_model, trend_features = joblib.load("trend_model.pkl")
vol_bundle = joblib.load("volatility_model.pkl")

vol_model = vol_bundle["model"]
vol_features = vol_bundle["feature_cols"]
regime_model, regime_features = joblib.load("regime_model.pkl")


data = pd.read_csv("Feature_engineered_data.csv")

with open("sentiment_output.json") as f:
    sentiment = json.load(f)

sentiment = pd.DataFrame([sentiment])

sentiment_scores = {
    "final_combined_score": float(sentiment["final_combined_score"].iloc[0]),
    "company_sentiment_score": float(sentiment["company_sentiment_score"].iloc[0]),
    "market_sentiment_score": float(sentiment["market_sentiment_score"].iloc[0]),
    "macro_market_sentiment": float(sentiment["macro_market_sentiment"].iloc[0])
}

data["trend_pred"] = trend_model.predict(data[trend_features])
data["volatility_pred"] = vol_model.predict(data[vol_features])
data["regime_pred"] = regime_model.predict(data[regime_features])

latest = data.iloc[-1]
combined_signal = {
    "trend_prediction": int(latest["trend_pred"]),
    "volatility_prediction": int(latest["volatility_pred"]),
    "regime_prediction": int(latest["regime_pred"]),
    "sentiment_score": sentiment_scores
  
}

import json

combined_json = json.dumps(combined_signal, indent=2)

print(combined_json)

#final_combined_score:
#Overall sentiment for the company after combining company news, market sentiment, and momentum. Range: -1 (very bearish) to +1 (very bullish).

#ompany_sentiment_score:
#Sentiment from news directly about the company using FinBERT. Range: -1 (very negative) to +1 (very positive).

#market_sentiment_score:
#Overall market environment sentiment derived from macro and sector news. Range: -1 (bearish market) to +1 (bullish market).

#macro_market_sentiment:
#Sentiment from macroeconomic news such as inflation, interest rates, GDP, and central bank policy. Range: -1 (negative outlook) to +1 (positive outlook).

#trend_prediction:
#Direction forecast for the next trading day. 
#1 → price expected to rise (bullish trend)
#0 → neutral/uncertain trend
#-1 → price expected to fall (bearish trend)

#volatility_prediction:
#Log of predicted standard deviation of returns for the next 5 days.
#Actual volatility = e^(value). 
#Example: -4 → e^-4 ≈ 0.018 ≈ 1.8% expected volatility.

#regime_prediction:
#Market regime classification based on market conditions.
#0 → Bear regime (downtrend / risk-off)
#1 → Sideways regime (range-bound market)
#2 → Bull regime (uptrend / risk-on)