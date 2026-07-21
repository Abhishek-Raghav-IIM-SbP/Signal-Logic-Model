import pandas as pd
import numpy as np

def engineer_features(input_csv, output_csv):
    
    # ==============================
    # 1️⃣ Load Raw Data
    # ==============================
    df = pd.read_csv(input_csv)

    # Make sure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Date"] = df["Date"].dt.tz_convert(None)

    # ==============================
    # 2️⃣ Returns
    # ==============================
    df["Returns"] = df["Close"].pct_change()

    # ==============================
    # 3️⃣ EMA 20 / 50 / 200
    # ==============================
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()


    df["Rolling_Vol_5"] = df["Returns"].rolling(5).std().shift(1)
    df["Rolling_Vol_10"] = df["Returns"].rolling(10).std().shift(1)
    df["ATR_proxy"] = (df["High"] - df["Low"]).shift(1)
    df["Abs_Return_lag1"] = df["Returns"].abs().shift(1)

    # ==============================
    # 4️⃣ RSI (14)
    # ==============================
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ==============================
    # 5️⃣ MACD
    # ==============================
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ==============================
    # 6️⃣ ADX
    # ==============================
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.rolling(14).mean()

    # ==============================
    # 7️⃣ Momentum (10)
    # ==============================
    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    # ==============================
    # 8️⃣ Volume Ratio
    # ==============================
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]
    

    # ==============================
    # 9️⃣ Breakout Indicators
    # ==============================
    df["20Day_High"] = df["High"].rolling(20).max()
    df["20Day_Low"] = df["Low"].rolling(20).min()

    df["Breakout_Up"] = (df["Close"] > df["20Day_High"].shift(1)).astype(int)
    df["Breakout_Down"] = (df["Close"] < df["20Day_Low"].shift(1)).astype(int)

    # ==============================
    # 10️⃣ Rolling Sharpe Ratio 20 days
    # ==============================
    rolling_sharpe = df["Returns"].rolling(20).mean() / df["Returns"].rolling(20).std()
    df["Rolling_Sharpe_20"] = rolling_sharpe
    
    
    #start 
    volatility = df["Returns"].rolling(20).std()

     # Rolling Sharpe Ratio (shifted to avoid leakage)
    rolling_mean = df["Returns"].rolling(20).mean().shift(1)
    rolling_std = df["Returns"].rolling(20).std().shift(1)
    df["Rolling_Sharpe_20"] = rolling_mean / rolling_std

    df["Target"] = (df["Returns"].shift(-1) > volatility*0.3).astype(int)
    lag_periods = [1, 2, 3]

    lag_cols = [

        "Returns",
        "EMA_20",
        "EMA_50",
        "EMA_200",

        "RSI",

        "MACD",
        "MACD_Signal",

        "ADX",

        "Momentum",

        "Volume_Ratio",

        "Breakout_Up",
        "Breakout_Down",
        "Rolling_Sharpe_20"

    ]

    for col in lag_cols:
        for lag in lag_periods:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
   
    # Trend alignment feature (VERY POWERFUL)
    df["Trend_alignment"] = (
        (df["EMA_20"] > df["EMA_50"]) &
        (df["EMA_50"] > df["EMA_200"])
    ).astype(int)

    df["Trend_alignment_lag1"] = df["Trend_alignment"].shift(1)
    df["Trend_alignment_lag2"] = df["Trend_alignment"].shift(2)
    df["RSI_change"] = df["RSI"] - df["RSI"].shift(1)

    df["RSI_change_lag1"] = df["RSI_change"].shift(1)
    df["RSI_change_lag2"] = df["RSI_change"].shift(2)
    df = df.dropna().reset_index(drop=True)
    feature_cols = []

    for col in lag_cols:
        for lag in lag_periods:
            feature_cols.append(f"{col}_lag{lag}")
    #end 
    # val features 
    df["Rolling_Vol_5"] = df["Returns"].rolling(5).std().shift(1)
    df["Rolling_Vol_10"] = df["Returns"].rolling(10).std().shift(1)
    df["ATR_proxy"] = (df["High"] - df["Low"]).shift(1)
    df["Abs_Return_lag1"] = df["Returns"].abs().shift(1)
    df["Vol_Ratio"] = df["Rolling_Vol_5"] / (df["Rolling_Vol_10"] + 1e-6)
    df["Vol_Change"] = df["Rolling_Vol_5"] - df["Rolling_Vol_10"]
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    df["Next_5D_Volatility"] = (
        df["Returns"].rolling(5).std().shift(-5)
    )
    df["Next_5D_Volatility"] = np.log(df["Next_5D_Volatility"])

    # Lagged volatility features (based on the log target)
    df["Vol_lag1"] = df["Next_5D_Volatility"].shift(1)
    df["Vol_lag2"] = df["Next_5D_Volatility"].shift(2)
    #vol features end
    
    
    # ==============================
    # 🔟 Optional: Target Variable (Next Day Direction)
    # ==============================
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Remove NaNs created by rolling calculations
    df.dropna(inplace=True)

    # ==============================
    # Save Processed CSV
    # ==============================
    df.to_csv(output_csv, index=False)

    print(f"Feature engineered data saved to {output_csv}")


engineer_features("Raw_data.csv", "Feature_engineered_data.csv")
engineer_features("Market_data.csv", "Feature_engineered_market_data.csv")
  # Example usage