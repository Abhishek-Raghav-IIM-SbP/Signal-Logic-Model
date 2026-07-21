
from sentiment_predictor import analyze_company_sentiment
import json
ticker="TCS"
result = analyze_company_sentiment(ticker, days=30, finnhub_api_key="d6jk8o1r01qkvh5q9a1gd6jk8o1r01qkvh5q9a20")
filename = "sentiment_output.json"

with open(filename, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(result)