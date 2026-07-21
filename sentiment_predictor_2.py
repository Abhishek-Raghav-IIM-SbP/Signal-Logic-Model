#!/usr/bin/env python3
"""
Dual-layer sentiment engine using Finnhub and FinBERT.

Computes:
- Company sentiment score
- Macro market sentiment score
- Sector sentiment score
- Aggregate market sentiment score
- Sentiment momentum
- Final combined score
- Classification and confidence

CLI:
    python sentiment_predictor.py --ticker RELIANCE.NS --days 7
"""

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import finnhub  # type: ignore[import]
import numpy as np
import pandas as pd
import requests
import torch  # type: ignore[import]
from transformers import (  # type: ignore[import]
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# -----------------------------
# Constants
# -----------------------------

MACRO_KEYWORDS: List[str] = [
    "indian stock market",
    "indian equities",
    "sensex",
    "nifty",
    "rbi",
    "inflation",
    "interest rates",
    "gdp",
    "federal reserve",
    "global markets",
    "oil prices",
    "foreign investors",
    "fii flows",
]

SECTOR_TICKERS: List[str] = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "LT.NS",
]


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Article:
    """Container for a single news article and its sentiment attributes."""

    source: str  # "company" | "macro" | "sector" | "market"
    headline: str
    summary: str
    datetime_utc: datetime
    days_old: float
    score: float  # continuous sentiment in [-1, 1]
    weight: float  # final weight including time decay and importance


# -----------------------------
# Sentiment Engine
# -----------------------------


class SentimentEngine:
    """
    Dual-layer sentiment engine using Finnhub news and FinBERT.

    Primary API:
        engine = SentimentEngine(...)
        result = engine.run(ticker="RELIANCE.NS", company_name="Reliance", days=7)
    """

    FINBERT_MODEL_NAME = "ProsusAI/finbert"

    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        india_vix_threshold: float = 18.0,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize engine, Finnhub client and FinBERT model.

        :param finnhub_api_key: Finnhub API key (or read from FINNHUB_API_KEY env).
        :param india_vix_threshold: (retained for compatibility, not used in fusion now).
        :param device: "cpu" or "cuda"; if None, auto-detect.
        """
        key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        if not key:
            raise RuntimeError("FINNHUB_API_KEY environment variable is not set.")

        self.client = finnhub.Client(api_key=key)
        self.api_key = key
        self.india_vix_threshold = india_vix_threshold

        # Device selection for FinBERT
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained(self.FINBERT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.FINBERT_MODEL_NAME
        ).to(self.device)
        self.model.eval()

        # Label mapping for FinBERT (usually: 0=negative,1=neutral,2=positive)
        self.id2label: Dict[int, str] = self.model.config.id2label

    # -----------------------------
    # Generic helpers
    # -----------------------------

    def _date_range(self, days: int) -> Tuple[str, str]:
        """Return (from_date, to_date) as YYYY-MM-DD in UTC."""
        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(days=days)
        return start_utc.date().isoformat(), now_utc.date().isoformat()

    # -----------------------------
    # Fetching: company, macro, sector, legacy market
    # -----------------------------

    def fetch_company_news(
        self, ticker: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Fetch company-specific news from Finnhub for a given ticker.

        :param ticker: Company ticker symbol.
        :param days: Lookback window in days.
        """
        from_date, to_date = self._date_range(days)
        try:
            news = self.client.company_news(ticker, _from=from_date, to=to_date) or []
        except Exception:
            news = []
        return news

    def fetch_market_news(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Legacy market news fetcher (kept for compatibility, not used in scoring).

        Uses NIFTY-related index symbols and general news with broad filters.
        """
        from_date, to_date = self._date_range(days)
        all_news: List[Dict[str, Any]] = []

        index_symbols = ["^NSEI", "NSEI", "NIFTY", "^NSEI:IND"]
        for symbol in index_symbols:
            try:
                news = self.client.company_news(symbol, _from=from_date, to=to_date)
            except Exception:
                news = []
            all_news.extend(news or [])

        base_url = "https://finnhub.io/api/v1/news"
        try:
            params = {"category": "general", "token": self.api_key}
            resp = requests.get(base_url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json() or []
        except Exception:
            data = []
        all_news.extend(data)

        return all_news

    def fetch_macro_news(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch macro-level market news using Finnhub's general news endpoint
        and filter by MACRO_KEYWORDS.

        Returns a list of cleaned article dicts compatible with `clean_articles`.
        """
        base_url = "https://finnhub.io/api/v1/news"
        try:
            params = {"category": "general", "token": self.api_key}
            resp = requests.get(base_url, params=params, timeout=5)
            resp.raise_for_status()
            raw = resp.json() or []
        except Exception:
            raw = []

        if not raw:
            return []

        keywords_lower = [k.lower() for k in MACRO_KEYWORDS]
        filtered: List[Dict[str, Any]] = []

        for item in raw:
            headline = (item.get("headline") or "").lower()
            summary = (item.get("summary") or "").lower()
            combined = f"{headline} {summary}"
            if any(kw in combined for kw in keywords_lower):
                filtered.append(item)

        # Reuse cleaning logic
        return self.clean_articles(filtered, source="macro", days=days)

    def fetch_sector_news(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch sector-level sentiment proxy by aggregating news for a set of
        major index companies (SECTOR_TICKERS).

        Returns a list of cleaned article dicts compatible with `clean_articles`.
        """
        from_date, to_date = self._date_range(days)
        raw: List[Dict[str, Any]] = []

        for ticker in SECTOR_TICKERS:
            try:
                news = self.client.company_news(ticker, _from=from_date, to=to_date)
            except Exception:
                news = []
            raw.extend(news or [])

        return self.clean_articles(raw, source="sector", days=days)

    # -----------------------------
    # Cleaning & preprocessing
    # -----------------------------

    @staticmethod
    def clean_articles(
        raw_articles: List[Dict[str, Any]],
        source: str,
        days: int,
    ) -> List[Dict[str, Any]]:
        """
        Clean raw Finnhub articles:
        - Remove duplicate headlines
        - Drop missing or very short text
        - Enforce lookback window (safety)
        - Preserve timezone-aware datetime
        """
        if not raw_articles:
            return []

        now_utc = datetime.now(timezone.utc)
        seen_headlines = set()
        cleaned: List[Dict[str, Any]] = []

        for item in raw_articles:
            headline = (item.get("headline") or "").strip()
            summary = (item.get("summary") or "").strip()

            if not headline or not summary:
                continue

            text = f"{headline}. {summary}"
            if len(text) < 40:
                continue

            if headline in seen_headlines:
                continue
            seen_headlines.add(headline)

            ts = item.get("datetime") or item.get("time") or item.get("publishedTime")
            if ts is None:
                continue

            try:
                dt_utc = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            except Exception:
                continue

            age_days = (now_utc - dt_utc).total_seconds() / 86400.0
            if age_days < 0 or age_days > days + 1:
                continue

            cleaned.append(
                {
                    "source": source,
                    "headline": headline,
                    "summary": summary,
                    "datetime_utc": dt_utc,
                    "days_old": age_days,
                }
            )

        return cleaned

    # -----------------------------
    # Sentiment calculation (FinBERT)
    # -----------------------------

    def compute_article_sentiment(self, text: str) -> float:
        """
        Compute continuous sentiment score in [-1, 1] for article text using FinBERT.

        score = P(positive) - P(negative)
        """
        if not text:
            return 0.0

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        label_probs: Dict[str, float] = {}
        for idx, p in enumerate(probs):
            label = self.id2label.get(idx, "").lower()
            label_probs[label] = float(p)

        p_pos = label_probs.get("positive", 0.0)
        p_neg = label_probs.get("negative", 0.0)

        score = float(p_pos - p_neg)
        return max(min(score, 1.0), -1.0)

    # -----------------------------
    # Weighting logic
    # -----------------------------

    @staticmethod
    def _importance_weight(text: str) -> float:
        """
        Determine importance-based weight multiplier from text keywords.

        Base weight = 1.0, multiplied per matched category and clipped.
        """
        base = 1.0
        lower = text.lower()

        if "earning" in lower or "results" in lower:
            base *= 2.5

        if "upgrade" in lower or "downgrade" in lower or "rating" in lower:
            base *= 2.0

        if "regulator" in lower or "regulatory" in lower or "sebi" in lower:
            base *= 2.0

        if "rbi" in lower or "gdp" in lower or "inflation" in lower or "macro" in lower:
            base *= 1.5

        return float(min(base, 5.0))

    @staticmethod
    def _time_decay_weight(days_old: float) -> float:
        """Exponential time decay: exp(-0.3 * days_old)."""
        days_old = max(days_old, 0.0)
        return float(math.exp(-0.3 * days_old))

    def _build_article_objects(
        self, cleaned: List[Dict[str, Any]]
    ) -> List[Article]:
        """
        Convert cleaned article dicts into `Article` objects with
        FinBERT sentiment scores and final weights applied.
        """
        articles: List[Article] = []
        for item in cleaned:
            headline = item["headline"]
            summary = item["summary"]
            dt_utc = item["datetime_utc"]
            days_old = float(item["days_old"])

            text = f"{headline}. {summary}"
            score = self.compute_article_sentiment(text)

            importance = self._importance_weight(text)
            time_w = self._time_decay_weight(days_old)
            weight = importance * time_w

            if weight <= 0.0:
                continue

            articles.append(
                Article(
                    source=item["source"],
                    headline=headline,
                    summary=summary,
                    datetime_utc=dt_utc,
                    days_old=days_old,
                    score=score,
                    weight=weight,
                )
            )

        return articles

    # -----------------------------
    # Aggregate scores
    # -----------------------------

    @staticmethod
    def compute_weighted_score(articles: List[Article]) -> Tuple[float, int]:
        """
        Compute weighted sentiment score:

            score = Σ(score * weight) / Σ(weight)

        Returns (score, total_articles).
        """
        if not articles:
            return 0.0, 0

        weights = np.array([a.weight for a in articles], dtype=float)
        scores = np.array([a.score for a in articles], dtype=float)
        denom = weights.sum()
        if denom <= 0:
            return 0.0, len(articles)

        weighted_score = float((scores * weights).sum() / denom)
        weighted_score = max(min(weighted_score, 1.0), -1.0)
        return weighted_score, len(articles)

    @staticmethod
    def compute_momentum(all_articles: List[Article]) -> float:
        """
        Compute sentiment momentum as:

            momentum = today_score - 3_day_average_score

        where scores are daily weighted sentiment across all articles
        (company + macro + sector). If there are fewer than 4 days,
        momentum is 0.
        """
        if not all_articles:
            return 0.0

        rows = []
        for a in all_articles:
            day = a.datetime_utc.date()
            rows.append(
                {
                    "date": day,
                    "score": a.score,
                    "weight": a.weight,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return 0.0

        grouped = df.groupby("date").apply(
            lambda g: np.average(g["score"], weights=g["weight"])
        )
        grouped = grouped.sort_index()

        if len(grouped) < 4:
            return 0.0

        today_score = float(grouped.iloc[-1])
        prev_three = float(grouped.iloc[-4:-1].mean())

        momentum = today_score - prev_three
        return float(max(min(momentum, 1.0), -1.0))

    # -----------------------------
    # Final fusion
    # -----------------------------

    def combine_scores(
        self,
        company_score: float,
        market_score: float,
        momentum: float,
    ) -> Tuple[float, str]:
        """
        Combine company & market sentiment plus momentum into final score.

        Final formula (additive fusion):

            final_score =
                (0.65 * company_sentiment_score)
              + (0.35 * market_sentiment_score)
              + (0.15 * sentiment_momentum)

        Returned score is clipped to [-1, 1], and classified as
        Bullish / Bearish / Neutral based on thresholds.
        """
        final_score = (
            0.65 * company_score
            + 0.35 * market_score
            + 0.15 * momentum
        )
        final_score = max(min(final_score, 1.0), -1.0)

        if final_score > 0.3:
            label = "Bullish"
        elif final_score < -0.3:
            label = "Bearish"
        else:
            label = "Neutral"

        return final_score, label

    @staticmethod
    def compute_confidence(final_score: float, total_articles: int) -> str:
        """
        Confidence rules:

        High:
            - articles >= 12
            - abs(score) > 0.6

        Medium:
            - articles >= 5

        Else:
            Low
        """
        if total_articles >= 12 and abs(final_score) > 0.6:
            return "High"
        if total_articles >= 5:
            return "Medium"
        return "Low"

    # -----------------------------
    # Public API
    # -----------------------------

    def run(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Run full sentiment pipeline for a ticker.

        :param ticker: Company ticker symbol.
        :param company_name: Optional human-readable company name.
        :param days: Lookback window in days.
        :return: Dict with sentiment metrics, ready for ML pipelines.
        """
        company_name = company_name or ticker

        # 1) Fetch & clean news
        raw_company = self.fetch_company_news(ticker, days=days)
        raw_macro_clean = self.fetch_macro_news(days=days)
        raw_sector_clean = self.fetch_sector_news(days=days)

        company_clean = self.clean_articles(raw_company, "company", days)

        # Macro & sector are already cleaned by their respective fetchers
        macro_clean = raw_macro_clean
        sector_clean = raw_sector_clean

        # 2) Build article objects with sentiment and weights
        company_articles = self._build_article_objects(company_clean)
        macro_articles = self._build_article_objects(macro_clean)
        sector_articles = self._build_article_objects(sector_clean)

        # 3) Compute layer-wise scores
        company_score, n_company = self.compute_weighted_score(company_articles)
        macro_score, n_macro = self.compute_weighted_score(macro_articles)
        sector_score, n_sector = self.compute_weighted_score(sector_articles)

        # 4) Market sentiment from macro + sector
        if n_macro == 0 and n_sector == 0:
            market_score = 0.0
        else:
            market_score = 0.6 * macro_score + 0.4 * sector_score
            market_score = max(min(market_score, 1.0), -1.0)

        # 5) Momentum from all articles
        all_articles = company_articles + macro_articles + sector_articles
        momentum = self.compute_momentum(all_articles)

        # 6) Fusion
        final_score, classification = self.combine_scores(
            company_score, market_score, momentum
        )

        # 7) Confidence
        total_articles = n_company + n_macro + n_sector
        confidence = self.compute_confidence(final_score, total_articles)

        result: Dict[str, Any] = {
            "company_sentiment_score": float(company_score),
            "macro_market_sentiment": float(macro_score),
            "sector_sentiment_score": float(sector_score),
            "market_sentiment_score": float(market_score),
            "sentiment_momentum": float(momentum),
            "final_combined_score": float(final_score),
            "classification": classification,
            "confidence_level": confidence,
            "total_company_articles": int(n_company),
            "total_macro_articles": int(n_macro),
            "total_sector_articles": int(n_sector),
        }

        return result


# -----------------------------
# Convenience function for other modules
# -----------------------------


def analyze_company_sentiment(
    company_name: str,
    days: int = 7,
    finnhub_api_key: Optional[str] = None,
    india_vix_threshold: float = 18.0,
) -> Dict[str, Any]:
    """
    Convenience function for other modules.

    Call with just a company name and lookback duration; the same string
    is used as the ticker symbol by default. Returns the sentiment metrics dict.
    """
    engine = SentimentEngine(
        finnhub_api_key=finnhub_api_key,
        india_vix_threshold=india_vix_threshold,
    )
    return engine.run(ticker=company_name, company_name=company_name, days=days)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-layer sentiment engine using Finnhub and FinBERT."
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Company ticker symbol, e.g. RELIANCE.NS",
    )
    parser.add_argument(
        "--company-name",
        required=False,
        default=None,
        help="Optional human-readable company name.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Lookback window in days (default: 7).",
    )
    parser.add_argument(
        "--vix-threshold",
        type=float,
        default=18.0,
        help="India VIX threshold (retained for compatibility, not used in fusion).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = SentimentEngine(india_vix_threshold=args.vix_threshold)
    result = engine.run(
        ticker=args.ticker,
        company_name=args.company_name,
        days=args.days,
    )

    import json

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()