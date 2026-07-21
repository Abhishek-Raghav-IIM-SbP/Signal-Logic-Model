import yfinance as yf
import pandas as pd

def fetch_and_save_to_csv(company, time_period):
    """
    company: ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
    time_period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'
    """
    # Append .NS for Indian stocks if no suffix is provided
    if not company.endswith('.NS') and not company.endswith('.BO') and company != 'GOOGL':
        company_yf = company + '.NS'
    else:
        company_yf = company

    # Fetch data
    stock = yf.Ticker(company_yf)
    data = stock.history(period=time_period)

    if data.empty:
        print(f"No data found for {company_yf}. Check ticker symbol.")
        return

    # Reset index so Date becomes a column
    data.reset_index(inplace=True)

    # Create filename automatically
    filename = "Raw_data.csv"

    # Save to CSV
    data.to_csv(filename, index=False)

    print(f"Data for {company_yf} saved successfully as {filename}")

def fetch_and_save_market_data(time_period):
    """
    Fetch and save market data
    """
    market = yf.Ticker('^NSEI')
    data = market.history(period=time_period)
    data.reset_index(inplace=True)
    data.to_csv('Market_data.csv', index=False)
    print("Market data saved successfully as Market_data.csv")

if __name__ == "__main__":
    fetch_and_save_to_csv('TCS', '2y')
    fetch_and_save_market_data('2y')