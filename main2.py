import joblib
import pandas as pd
import json
import requests

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"   # <-- replace with your key
GROQ_MODEL   = "llama-3.3-70b-versatile"  # or: mixtral-8x7b-32768, llama-3.1-8b-instant

# ── Predefined system prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert financial market analyst and trading strategist.
You will receive a JSON object containing:
final_combined_score:
Overall sentiment for the company after combining company news, market sentiment, and momentum. Range: -1 (very bearish) to +1 (very bullish).

company_sentiment_score:
Sentiment from news directly about the company using FinBERT. Range: -1 (very negative) to +1 (very positive).

market_sentiment_score:
Overall market environment sentiment derived from macro and sector news. Range: -1 (bearish market) to +1 (bullish market).

macro_market_sentiment:
Sentiment from macroeconomic news such as inflation, interest rates, GDP, and central bank policy. Range: -1 (negative outlook) to +1 (positive outlook).

trend_prediction:
Direction forecast for the next trading day. 
1 → price expected to rise (bullish trend)
0 → neutral/uncertain trend
-1 → price expected to fall (bearish trend)

volatility_prediction:
Log of predicted standard deviation of returns for the next 5 days.
Actual volatility = e^(value). 
Example: -4 → e^-4 ≈ 0.018 ≈ 1.8% expected volatility.

regime_prediction:
Market regime classification based on market conditions.
0 → Bear regime (downtrend / risk-off)
1 → Sideways regime (range-bound market)
2 → Bull regime (uptrend / risk-on)

Based on this data, provide:
clear recommendation of BUY / SELL / HOLD with confidence level (Low / Medium / High) in square brackets separately 

1. A clear market analysis summary (2-3 sentences)
2. Key risk factors to watch
3. A short reasoning paragraph justifying your recommendation


Be concise, data-driven, and avoid generic advice.
""".strip()

# ── Load models ────────────────────────────────────────────────────────────────
trend_model,  trend_features  = joblib.load("trend_model.pkl")
vol_bundle                     = joblib.load("volatility_model.pkl")
vol_model                      = vol_bundle["model"]
vol_features                   = vol_bundle["feature_cols"]
regime_model, regime_features  = joblib.load("regime_model.pkl")

# ── Load data ──────────────────────────────────────────────────────────────────
data = pd.read_csv("Feature_engineered_data.csv")

with open("sentiment_output.json") as f:
    sentiment = json.load(f)

sentiment = pd.DataFrame([sentiment])
sentiment_scores = {
    "final_combined_score":     float(sentiment["final_combined_score"].iloc[0]),
    "company_sentiment_score":  float(sentiment["company_sentiment_score"].iloc[0]),
    "market_sentiment_score":   float(sentiment["market_sentiment_score"].iloc[0]),
    "macro_market_sentiment":   float(sentiment["macro_market_sentiment"].iloc[0]),
}

# ── Run model predictions ──────────────────────────────────────────────────────
data["trend_pred"]     = trend_model.predict(data[trend_features])
data["volatility_pred"] = vol_model.predict(data[vol_features])
data["regime_pred"]    = regime_model.predict(data[regime_features])

latest = data.iloc[-1]
combined_signal = {
    "trend_prediction":      int(latest["trend_pred"]),
    "volatility_prediction": float(latest["volatility_pred"]),
    "regime_prediction":     int(latest["regime_pred"]),
    "sentiment_score":       sentiment_scores,
}

# ── Send to Groq LLM ───────────────────────────────────────────────────────────
def get_llm_analysis(signal: dict) -> str:
    user_message = f"Here is the current market signal data:\n\n{json.dumps(signal, indent=2)}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.3,   # lower = more consistent/deterministic output
        "max_tokens":  1024,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json=payload,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Combined Signal ──────────────────────────────────────")
    print(json.dumps(combined_signal, indent=2))

    print("\n── LLM Analysis ─────────────────────────────────────────")
    analysis = get_llm_analysis(combined_signal)
    print(analysis)

    with open("analysis_output.txt", "w") as f:
        f.write(analysis)
    print("✅ Analysis saved to analysis_output.txt")