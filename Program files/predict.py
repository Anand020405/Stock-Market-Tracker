import torch
import os
import yfinance as yf
from training import Model


def predict(stock_index, timeline = '1d'):
    model = Model()
    if os.path.exists(f"{stock_index}.pth"):
        model = torch.load(f"{stock_index}.pth", weights_only=False)
        model.eval()
    else:
        raise MemoryError

    stock_data_ticker = yf.Ticker(stock_index)
    data = stock_data_ticker.history(timeline)
    Volume = list(data['Volume'])
    Open = list(data['Open'])
    High = list(data['High'])
    Low = list(data['Low'])
    Close = list(data['Close'])
    data = [Open[0], High[0], Low[0], Volume[0], Close[0]]
    prediction_data = torch.Tensor(data)
    with torch.no_grad():
        down,neutral,up = model(prediction_data)
    if int(down) >= int(neutral):
        return "Down"
    else:
        if int(neutral) <= int(up):
            return "Up"
        else:
            return "Neutral"

if __name__ == "__main__":
    predict("TCS.NS")