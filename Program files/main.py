import functions

stock_name = input("Enter the stock name that you want to search for: ")
time_period = input("Enter the time period you want to display: ")

stock_data = functions.get_stock_data(stock_name,time_period)
functions.display_graph(stock_data)
print(functions.analyst_price_targets(stock_name))