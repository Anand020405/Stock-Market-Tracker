import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


STOCK_NAME_LIST_DATA = {
        "Adani Enterprises": "ADANIENT",
        "Adani Ports & SEZ": "ADANIPORTS",
        "Apollo Hospitals": "APOLLOHOSP",
        "Asian Paints": "ASIANPAINT",
        "Axis Bank": "AXISBANK",
        "Bajaj Auto": "BAJAJ-AUTO",
        "Bajaj Finance": "BAJFINANCE",
        "Bajaj Finserv": "BAJAJFINSV",
        "Bharat Electronics": "BEL",
        "Bharat Petroleum": "BPCL",
        "Bharti Airtel": "BHARTIARTL",
        "Britannia Industries": "BRITANNIA",
        "Cipla": "CIPLA",
        "Coal India": "COALINDIA",
        "Dr. Reddy's Laboratories": "DRREDDY",
        "Eicher Motors": "EICHERMOT",
        "Grasim Industries": "GRASIM",
        "HCLTech": "HCLTECH",
        "HDFC Bank": "HDFCBANK",
        "HDFC Life": "HDFCLIFE",
        "Hero MotoCorp": "HEROMOTOCO",
        "Hindalco Industries": "HINDALCO",
        "Hindustan Unilever": "HINDUNILVR",
        "ICICI Bank": "ICICIBANK",
        "IndusInd Bank": "INDUSINDBK",
        "Infosys": "INFY",
        "ITC": "ITC",
        "JSW Steel": "JSWSTEEL",
        "Kotak Mahindra Bank": "KOTAKBANK",
        "Larsen & Toubro": "LT",
        "Mahindra & Mahindra": "M&M",
        "Maruti Suzuki": "MARUTI",
        "Nestlé India": "NESTLEIND",
        "NTPC": "NTPC",
        "Oil and Natural Gas Corporation": "ONGC",
        "Power Grid": "POWERGRID",
        "Reliance Industries": "RELIANCE",
        "SBI Life Insurance Company": "SBILIFE",
        "Shriram Finance": "SHRIRAMFIN",
        "State Bank of India": "SBIN",
        "Sun Pharma": "SUNPHARMA",
        "Tata Consultancy Services": "TCS",
        "Tata Consumer Products": "TATACONSUM",
        "Tata Motors": "TATAMOTORS",
        "Tata Steel": "TATASTEEL",
        "Tech Mahindra": "TECHM",
        "Titan Company": "TITAN",
        "Trent": "TRENT",
        "UltraTech Cement": "ULTRACEMCO",
        "Wipro": "WIPRO"
    }

TIMELINES = ["1d","5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

def get_stock_data(stock_name, period):
    stock_index = get_stock_index(stock_name)
    stock_data_ticker = yf.Ticker(stock_index)
    stock_data = stock_data_ticker.history(period)
    return stock_data

def display_graph(stock_data):
    plt.plot(stock_data['Close'])
    plt.show()

def get_stock_index(stock_name):
    return STOCK_NAME_LIST_DATA[stock_name] + '.NS'

def analyst_price_targets(stock_name):
    stock_index = get_stock_index(stock_name)
    stock_data_ticker = yf.Ticker(stock_index)
    return stock_data_ticker.analyst_price_targets

def get_stock_names():
    return [stock_name for stock_name in STOCK_NAME_LIST_DATA.keys()]

def get_timeline():
    return TIMELINES

if __name__ == "__main__":
    stock_data = get_stock_data('Wipro','1y')
    list1 = [str(x).split()[0][5:] for x in stock_data.index]
    data = [x for x in stock_data['Close']]
    data = pd.Series(data,list1)
    print(data)
    plt.plot(data)
    plt.show()