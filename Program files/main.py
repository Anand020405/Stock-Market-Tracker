import functions
import FreeSimpleGUI as sg


'''
stock_name = input("Enter the stock name that you want to search for: ").capitalize()
time_period = input("Enter the time period you want to display: ").lower()

stock_data = functions.get_stock_data(stock_name,time_period)
functions.display_graph(stock_data)'''

stock_name_text = sg.Text("Stock: ")
timeline_text = sg.Text("Timeline: ")

stock_name_input = sg.InputText(key='stock_name')
timeline_input = sg.InputText(key='timeline')

stock_name_listbox = sg.Listbox(functions.get_stock_names(), size=(30, 5), key='stock_list')
timeline_listbox = sg.Listbox(functions.get_timeline(), size=(10, 5), key= 'timeline_list')

stock_data_button = sg.Button("Get Data")

layout = [
    [stock_name_text,stock_name_input,stock_name_listbox],
    [timeline_text,timeline_input, timeline_listbox],
    [stock_data_button]
    ]

window = sg.Window("Stock Market Tracker",layout=layout)
previous_stock = ""
previous_timeline = ""
while True:
    event, values = window.read(timeout=1000)
    print(event, values)
    if event == sg.WIN_CLOSED:
        break

    if values['stock_list'] != [] and values['stock_list'][0] != previous_stock:
        window['stock_name'].update(value=values['stock_list'][0])
        previous_stock = values['stock_list'][0]    

    if values['timeline_list'] != [] and values['timeline_list'][0] != previous_timeline:
        window['timeline'].update(value=values['timeline_list'][0])
        previous_timeline = values['timeline_list'][0]   

    if event == "Get Data":
        stock_name = values['stock_name']
        time_period = values['timeline']
        stock_data = functions.get_stock_data(stock_name,time_period)
        functions.display_graph(stock_data)

window.close()