from crewai_tools import BaseTool
import pandas as pd
import ta
import re
from io import StringIO


class TechnicalIndicatorCalculator(BaseTool):
  name: str = "TechnicalIndicatorCalculator"
  description: str = """
  This tool is designed to compute various technical indicators for stock market data. 
  It processes input argument received directly from the stock_price tool, which includes date, open, high, low, close, volume, dividends, and stock splits. 
  Don't put EOF inside string starting at row 0!
  Action Input must be like this -> Action Input: {\"argument\": stock_price_data}
  And the stock_price_data format should be csv.
  You have to provide sufficient stock_price_data to get meaningful indicators. the data point count must be at least 30.
  """
  def _run(self, argument: str) -> str:
    data = argument
    
    df = pd.read_csv(StringIO(data), parse_dates=['Date'])

    # Set the Date column as the index
    df.set_index('Date', inplace=True)


    sampled_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    sampled_df = calculate_indicators(sampled_df)

    # Create a single string output
    output = []
    output.append("\nCalculated Indicators:")
    output.append(sampled_df.tail().to_string())

    # Join all parts into a single string
    final_output = "\n".join(output)

    return final_output

# Define a function to calculate indicators
def calculate_indicators(df):
    # Drop rows with any NaN values in essential columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']).copy()

    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Momentum Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['Stochastic_Oscillator'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['Williams_%R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

    # Volatility Indicators
    #df['Bollinger_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    #df['Bollinger_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    #df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    # Volume Indicators
    #df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    #df['VROC'] = df['Volume'].pct_change(periods=12) * 100

    # Trend Indicators
    #df['Parabolic_SAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()

    return df

