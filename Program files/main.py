import functions
import FreeSimpleGUI as sg

space1 = sg.Text("              ")
space2 = sg.Text("              ")
stock_name_text = sg.Text("Stock:    ")
timeline_text = sg.Text("Timeline: ")
stock_info_text = sg.Text("",key="stock_info")

stock_name_input = sg.InputText(key='stock_name', size=500)
timeline_input = sg.InputText(key='timeline', size=500)

stock_name_listbox = sg.Listbox(functions.get_stock_names(), size=(500, 5), key='stock_list', enable_events=True)
timeline_listbox = sg.Listbox(functions.get_timeline(), size=(500, 5), key= 'timeline_list', enable_events=True)

show_graph_button = sg.Button("Show Graph", key="graph")
stock_data_button = sg.Button("Get Data", key="data")
layout = [
    [stock_name_text,stock_name_input],
    [space1,stock_name_listbox],
    [timeline_text,timeline_input], 
    [space2, timeline_listbox],
    [stock_data_button, show_graph_button],
    [stock_info_text]
    ]

width = 800
window = sg.Window("Stock Market Tracker",layout=layout, size=(width,width))
previous_stock = ""
previous_timeline = ""
previous_stock_name = ""
previous_timeline_text = ""
TimeOut = 50
while True:
    event, values = window.read(timeout=TimeOut)
    if event == sg.WIN_CLOSED:
        break  
    
    if values['stock_name'] != "" and values['stock_name'] != previous_stock_name:
        hint = values['stock_name'].title().strip()
        window["stock_list"].update(functions.sort_stock_names(hint))
        previous_stock_name = values['stock_name']

    if values['timeline'] != "" and values['timeline'] != previous_timeline_text:
        hint = values['timeline'].title().strip()
        window["timeline_list"].update(functions.sort_timeline(hint))
        previous_timeline_text = values['timeline']
    

    if values['stock_list'] != [] and values['stock_list'][0] != previous_stock:
        window['stock_name'].update(value=values['stock_list'][0])
        previous_stock = values['stock_list'][0]    

    if values['timeline_list'] != [] and values['timeline_list'][0] != previous_timeline:
        window['timeline'].update(value=values['timeline_list'][0])
        previous_timeline = values['timeline_list'][0]   

    if event == "graph":
        try:
            stock_name = values['stock_name'].strip()
            time_period = values['timeline'].strip()
            print(time_period)
            if time_period == "":
                time_period = 'max'
            if functions.check_valid_time_period(time_period):
                stock_data = functions.get_stock_data_for_graph(stock_name,time_period)
                functions.display_graph(stock_data)
            else:
                sg.popup("Enter a Valid Timeline")
        
        except KeyError:
            sg.popup("Enter a Valid Stock Name")
        
    if event == "data":
        try:
            stock_name = values['stock_name']
            stock_info = functions.get_stock_info(stock_name)
            window["stock_info"].update(value=stock_info)
        except KeyError:
            sg.popup("Enter a Valid Stock Name")
window.close()