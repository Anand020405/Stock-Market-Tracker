import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
WIDTH = 100
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

TIMELINES = ["5d", "1mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

def get_stock_data_for_graph(stock_name, period):
    '''
        This function helps in getting the stock data in order to develop a graph for it
    '''
    stock_index = get_stock_index(stock_name)
    stock_data_ticker = yf.Ticker(stock_index)
    stock_data = stock_data_ticker.history(period)
    return stock_data

def display_graph(stock_data):
    '''
        This function is used to display the graph using the obtained stock data
    '''
    plt.plot(stock_data['Close'])
    plt.show()

def get_stock_index(stock_name):
    '''
        This function is used to retrieve all the stock index using the names of the stock
    '''
    return STOCK_NAME_LIST_DATA[stock_name] + '.NS'

def get_stock_names():
    '''
        This function is used to retrieve the stock names
    '''
    return [stock_name for stock_name in STOCK_NAME_LIST_DATA.keys()]

def get_timeline():
    '''
        This function helps in retriving all the available timeline for graph creation
    '''
    return TIMELINES

def get_stock_info(stock_name):
    '''
        This functions retrives all the available info of the stock and returns it in the form of a string
    '''
    stock_index = get_stock_index(stock_name)
    stock_data_ticker = yf.Ticker(stock_index)
    sector = wrap_text(stock_data_ticker.info['sector']).capitalize()
    long_business_summary = wrap_text(stock_data_ticker.info['longBusinessSummary']).capitalize()
    previous_close = wrap_text(stock_data_ticker.info['previousClose']).capitalize()
    open = wrap_text(stock_data_ticker.info['open']).capitalize()
    day_low = wrap_text(stock_data_ticker.info['dayLow']).capitalize()
    day_high = wrap_text(stock_data_ticker.info['dayHigh']).capitalize()
    dividend_yield = wrap_text(stock_data_ticker.info['dividendYield']).capitalize()
    price_to_book = wrap_text(stock_data_ticker.info['priceToBook']).capitalize()
    current_price = wrap_text(stock_data_ticker.info['currentPrice']).capitalize()
    target_high_price = wrap_text(stock_data_ticker.info['targetHighPrice']).capitalize()
    target_low_price = wrap_text(stock_data_ticker.info['targetLowPrice']).capitalize()
    recommendation_key = wrap_text(stock_data_ticker.info['recommendationKey']).capitalize()
    number_of_analyst_opinion = wrap_text(stock_data_ticker.info['numberOfAnalystOpinions']).capitalize()

    
    return_string = f"Sector: {sector} \nBusiness Summary: {long_business_summary}\nPrevious Close: {previous_close}\nOpen: {open}\nDay Low: {day_low}\nDay High: {day_high}\nDividend Yield: {dividend_yield}\nPrice to Book Value: {price_to_book}\nCurrent Price: {current_price}\nTarget High Price: {target_high_price}\nTarget Low Price: {target_low_price}\nRecommendation: {recommendation_key}\nNumber of Analyst: {number_of_analyst_opinion}\n"

    return return_string

def wrap_text(text):
    """
        This functions helps to convert a long string into a reasonable size for displaying on the screen
    """
    if type(text) != str:
        text = str(text)
    text_list = text.split()
    length = 0
    text = ""
    for word in text_list:
        length += len(word)
        if length > WIDTH:
            text += "\n\t"+word+" "
            length = len(word)
        else:
            text += word + " "
    return text

def check_valid_time_period(time_period):
    """
        This function checks if the input timeline is valid or not
    """
    if time_period in TIMELINES:
        return True
    return False

def sort_stock_names(hint):
    """
        This function helps in sorting the stock names based on the letters typed
    """
    stocks = get_stock_names()
    stocks_matching_hint = [stock for stock in stocks if stock.startswith(hint)]
    rest_of_stocks = stocks.copy()
    for stock in stocks_matching_hint:
        rest_of_stocks.remove(stock)
    sorted(rest_of_stocks)
    stocks_matching_hint.extend(rest_of_stocks)
    return stocks_matching_hint

def sort_timeline(hint):
    """
        This function helps in sorting the timeline based on the letters typed
    """
    timelines = get_timeline()
    timeline_matching_hint = [timeline for timeline in timelines if timeline.startswith(hint)]
    rest_of_timeline = timelines.copy()
    for timeline in timeline_matching_hint:
        rest_of_timeline.remove(timeline)
    sorted(rest_of_timeline)
    timeline_matching_hint.extend(rest_of_timeline)
    return timeline_matching_hint

if __name__ == "__main__":

    dat = yf.Ticker("MSFT")
    print(dat.quarterly_income_stmt)

