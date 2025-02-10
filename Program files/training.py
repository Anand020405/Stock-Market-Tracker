import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class Model(nn.Module):
    def __init__(self, in_features=5, h1=1000, h2=1000, h3=1000, h4=1000, h5=1000, h6=1000, h7=1000, h8=1000, h9=1000, h10=1000, h11=1000, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.fc6 = nn.Linear(h5, h6)
        self.fc7 = nn.Linear(h6, h7)
        self.fc8 = nn.Linear(h7, h8)
        self.fc9 = nn.Linear(h8, h9)
        self.fc10 = nn.Linear(h9, h10)
        self.fc11 = nn.Linear(h10, h11)
        self.out = nn.Linear(h11, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = self.out(x)
        return x

torch.manual_seed(41)

model = Model()

model = torch.load(r"C:\Users\bvana\Desktop\Apps\Stock Market Tracker\Program files\my_checkpoint1.pth", weights_only=False)
model.eval()

# Data
import yfinance as yf
import pandas as pd

stock_index = "TCS.NS"
stock_data_ticker = yf.Ticker(stock_index)
data = stock_data_ticker.history('max')
add_column = list(data['Close'])
add_column.insert(0,28.42286491394043)
add_column.pop()
data['PreviousClose'] = add_column

X = data.drop('Close', axis=1)
X = X.drop('Dividends', axis=1)
X = X.drop('Stock Splits', axis=1)
close = data['Close']
open = data['Open']
new_list = []
for x in data.index:
    if open[x] < close[x]:
        new_list.append(2)
    elif open[x] > close[x]:
        new_list.append(0)
    else:
        new_list.append(1)
data['Target'] = new_list
y = data['Target']


# Data End

X = np.round(X.values).astype(int)
y = np.round(y.values).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)


epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train) 
    losses.append(loss.detach().numpy())
    if i % 10 == 0:
        torch.save(model, r"C:\Users\bvana\Desktop\Apps\Stock Market Tracker\Program files\my_checkpoint1.pth")
        print(f'Epoch: {i} and loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test) 

print(f"Loss: ", loss)

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

    if y_val.argmax().item() == y_test[i]:
        correct +=1

print(f'We got {correct} correct out of {i}')

plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')