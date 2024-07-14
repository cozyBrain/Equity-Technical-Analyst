import os
from crewai import Agent , Crew , Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, tool, CodeInterpreterTool
from langchain.agents import Tool
from langchain_anthropic import ChatAnthropic
import yfinance as yf
from langchain_community.llms import Ollama
import pandas as pd
from tools.calculator_tools import TechnicalIndicatorCalculator

os.environ["DOCKER_HOST"] = f"unix:///run/user/{os.getuid()}/docker.sock"
# print all rows
pd.set_option('display.max_rows', None)
# print all columns
#pd.set_option('display.max_columns', None)

os.environ["ANTHROPIC_API_KEY"] = ''

ollama = Ollama(
    model = "orca-mini:7b",
    base_url = "http://localhost:11434")

custom_llm = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")

os.environ["SERPER_API_KEY"] = ""


#Tools

crawling_tool = SerperDevTool()
scraping_tool = ScrapeWebsiteTool()
code_interpreter_tool = CodeInterpreterTool()
technical_indicator_calculator_tool = TechnicalIndicatorCalculator()

@tool("Stock News")
def stock_news(ticker):
    """
    Useful to get news about a stock.
    The input should be a ticker. for example NVDA, 036460.KS.
    """
    ticker = yf.Ticker(ticker)
    return ticker.news
@tool("Stock Price")
def stock_price(ticker, start_date, end_date, interval):
    """
    Useful to get stock price data.
    The each args should be a ticker, start_date, end_date and interval.
    ticker examples: NVDA, 036460.KS.
    start and end date example: year-month-day
    interval examples: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    reason why this tool let you set start_date and end_date is to allow you to analyze separately in multiple times.
    When interval is 1 day, you can only get 2 months of price data at once to prevent overload. but you can get price data in multiple times to analyze more.
    When interval is 1 week, you can only get 1 year of price data at once to prevent overload.
    if you want to get latest data, the end_date must be 2 day later from current time. current time is 2024-7-14.
    You must Set Interval to 1 day when you get recent data that is less than a month old from stock price.
    """#1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    ticker = yf.Ticker(ticker)
    return ticker.history(start=start_date, end=end_date, interval=interval)
    #return ticker.history(period="max", interval="1wk")
    #yf.download("AAPL", start="2022-01-01", end="2022-12-31")
    #d = ticker.history(period="max", interval="3mo")
    #d = ticker.history(period="1y", interval='1wk')
@tool("Income Statement")
def income_stmt(ticker):
    """
    Useful to get the income statement of a company.
    The input should be a ticker. for example NVDA, 036460.KS.
    """
    ticker = yf.Ticker(ticker)
    return ticker.income_stmt
@tool("Balance Sheet")
def balance_sheet(ticker):
    """
    Useful to get the balance sheet of a company.
    The input should be a ticker. for example NVDA, 036460.KS.
    """
    ticker = yf.Ticker(ticker)
    return ticker.balance_sheet
@tool("Insider Transactions")
def inside_transactions(ticker):
    """
    Useful to get insider transactions of a stock.
    The input should be a ticker. for example NVDA, 036460.KS.
    """
    ticker = yf.Ticker(ticker)
    return ticker.insider_transactions

#python_programmer = Agent(
#    role='Python Programmer',
#    goal='build a financial analysis program in python code. execute the code using your tool.',
#    backstory="""
#    You can calculate technical analysis using python code. You can find many useful technical indicies from stock price history.
#    You are good at correcting the format and splitting the code into smaller chunks to avoid any issues.
#    """,
#    memory=True,
#    verbose=True,
#    allow_delegation=False,
#    tools=[code_interpreter_tool],
#    llm=custom_llm,
#    max_iter = 60,
#    rpm = 1,
#)

technical_analyst = Agent(
    role='Financial Analyst',
    goal='Provide every various financial analysis for {company} in detail. The financial analysis is based on latest data.',
    backstory="""
    you can only get 2 months of price data at once to prevent overload with stock price by day tool.
    You are good at technical analysis using technical index such as Williams %R, RSI, EMA, MACD, Stochastic Oscillator, Fibonacci, etc...
    You are used to collect stock price data in multiple times to prevent overload. max collecting periods at a time is 2months.
    You know better than anyone that not all data can be analyzed at the highest resolution, so you know how to set intervals appropriately.
    When Using stock_price:
        When Collecting 1 year older data, the interval must be 1 month.
        When Collecting data less than a year old data, the interval must be 1 week.
        When Collecting recent data that is less than a month You MUST Set Interval to 1 day.
    Using TechnicalIndicatorCalculator you will get some useful indicators.
    """,#You are used to work with the python programmer who help you find accurate technical values.
    memory=True,
    verbose=True,
    allow_delegation=True,
    tools=[stock_price, technical_indicator_calculator_tool],
    llm=custom_llm,
    max_iter = 60,
    rpm = 1,
)



technical_analysis_task_description = """
    "The final answer must provide everything!!"
    Analyze whole stock history(max periods).
    to analyze accurately, use python code!
    Technical analysis of {company} in weekly, monthly and yearly timeframes.
    Provide detailed support and resistance levels, trendlines, and chart patterns.
    Include analysis of the following technical indicators:

    1. Moving Averages(in weekly, monthly and yearly):
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)

    2. Momentum Indicators(in weekly, monthly and yearly):
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Stochastic Oscillator
    - Williams %R

    3. Volatility Indicators:
    - Bollinger Bands
    - Average True Range (ATR)

    4. Volume Indicators:
    - On-Balance Volume (OBV)
    - Volume Rate of Change (VROC)

    5. Trend Indicators:
    - Parabolic SAR
    - Average Directional Index (ADX)

    Provide the rate and direction of change across short, mid, long terms.
    And More:
"""
technical_analysis_task = Task(
    description=technical_analysis_task_description,
    expected_output=technical_analysis_task_description,
    agent=technical_analyst,
    output_file='outputs/technical_analysis.md',
    create_directory=True,
    async_execution=False,
)


#create Crewai

crew = Crew(
    agents=[
        technical_analyst,
        #python_programmer,
    ],
    tasks=[
        technical_analysis_task,
    ],
    verbose=2,
    max_rpm=2,
    #memory=True,
    output_log_file="crew_output_log.md"
)


result = crew.kickoff(inputs={'company': 'Apple', 'refs_to_read': 'How people and experties think about the company'})
print(result)
