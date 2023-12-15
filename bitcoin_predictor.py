"""
Bitcoin Price Prediction using GRU Neural Network
Initial implementation
"""

import pandas as pd
import numpy as np
import datetime
from binance.client import Client


class BitcoinPredictor:
    """Main class for Bitcoin price prediction"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize Bitcoin predictor"""
        self.client = Client(api_key, api_secret)
        
    def fetch_bitcoin_data(self, symbol: str = "BTCUSDT", 
                          start_date: str = "1 Jan 2019") -> pd.DataFrame:
        """Fetch Bitcoin historical data from Binance"""
        print(f"Fetching {symbol} data from {start_date}...")
        
        klines = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_date)
        bitcoin_df = pd.DataFrame(klines)
        
        bitcoin_df.columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote", "no_trades", "base_buy", "quote_buy", "ignore"
        ]
        
        # Convert timestamp to datetime
        bitcoin_df["date"] = bitcoin_df["open_time"].apply(
            lambda x: datetime.datetime.fromtimestamp(x/1000)
        )
        
        # Convert to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            bitcoin_df[col] = pd.to_numeric(bitcoin_df[col])
            
        print(f"Fetched {len(bitcoin_df)} records")
        return bitcoin_df


def main():
    """Main function"""
    predictor = BitcoinPredictor()
    df = predictor.fetch_bitcoin_data()
    print("Bitcoin data fetched successfully!")


if __name__ == "__main__":
    main() 