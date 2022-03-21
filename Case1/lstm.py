import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model_hpc import get_data

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill as pickle



class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, device = None):
        super().__init__()
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = default_device if device is None else device
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=self.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device = self.device).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        # out *= out > 0.0031622776601683794

        return out

class FlightDataset(Dataset):
    def __init__(self, type = "train", sequence_length = 5, device = None):
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = default_device if device is None else device
        self.sequence_length = sequence_length
        if type == "train":
            self.X = X_train
            self.y = y_train
        elif type == "validation":
            self.X = X_val
            self.y = y_val

        self.X = torch.tensor(self.X, device = device).float()
        self.y = torch.tensor(self.y, device = device).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,i):
        # return self.X[i], self.y[i]
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

def loss_function(yhat, y):
    deviance = (y-yhat)/y
    accuracy = (1-torch.abs(deviance))*100
    return -torch.mean(accuracy)


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        # X, y = X.to(device = device), y.to(device = device)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss

def test_model(data_loader, model, loss_function):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss

def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output.to(device), y_star.to(device)), 0)

    return output


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_format = "summary_stats" # summary_stats or onehot
    X, X_train, X_val, y, y_train, y_val = get_data(data_format)

    

    learning_rate = 5e-5
    num_hidden_units = [16, 32, 64]
    sequence_length = [range(1,10)]
    save_name = "models/lstm_model.pkl"

    i = 0
    N_models = len(sequence_length)*len(num_hidden_units)
    model_grid = {
        "num_hidden_units":[None]*N_models,
        "acc_val" : [None]*N_models,
        "acc_train" : [None]*N_models,
        "model" : [None]*N_models,
        "sequence_length" : [None]*N_models,
    }

    for sl in sequence_length:
        train_loader = DataLoader(FlightDataset(), batch_size=3, shuffle=True)
        test_loader = DataLoader(FlightDataset(type = "validation"), batch_size=3, shuffle=False)
        for hu in num_hidden_units:
            model = ShallowRegressionLSTM(num_sensors=X_train.shape[1], hidden_units=hu)

            print("Device: ", device)
            model.to(device)


            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            n_epochs = 20
            train_accs = np.zeros((n_epochs))
            test_accs = np.zeros((n_epochs))
            for ix_epoch in range(n_epochs):
                print(f"Epoch {ix_epoch}\n---------")
                train_accs[ix_epoch] = -train_model(train_loader, model, loss_function, optimizer=optimizer)
                test_accs[ix_epoch] = -test_model(test_loader, model, loss_function)
                print()

            # model_grid["lr"][i] = lr
            model_grid["num_hidden_units"][i] = hu
            model_grid["acc_train"][i] = train_accs
            model_grid["acc_val"][i] = test_accs
            model_grid["model"][i] = model
            model_grid["sequence_length"][i] = sl

            i += 1

            with open(save_name, "wb") as file:
                pickle.dump(model_grid, file)


    

