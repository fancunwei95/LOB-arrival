import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader


class OrderArrival_Dataset(Dataset):
    def __init__(self, data, window_size):
        super().__init__()
        
        self.window_size = window_size

        dates = list(set(data.index.floor("D")))
        dates.sort()
        self.dates = dates

        self.data = {date: data[data.index.floor("D") == date]
                    for date in dates}

        # remove last element since there is no y value for that entry (no +1 in next line)
        self.lengths = [ np.sum(self.data[date].index.floor("D") == date) - self.window_size
                for date in dates]


    def __len__(self):
        # remove last element since there is no y value for that entry
        return np.sum(self.lengths)

    def __getitem__(self, index):

        date = 0
        while (index >= self.lengths[date]):
            index = index - self.lengths[date]
            date +=1

        thisdata = self.data[self.dates[date]]
        idx = index + self.window_size -1
        Ask_seq = thisdata["Ask Arrival"].iloc[index:idx+1]
        Bid_seq = thisdata["Bid Arrival"].iloc[index:idx+1]

        past_seq = np.zeros((Ask_seq.shape[0],2))
        past_seq[:,0] = Ask_seq.to_numpy()
        past_seq[:,1] = Bid_seq.to_numpy()

        Ask_pred = thisdata["Ask pred"].iloc[idx]
        Bid_pred = thisdata["Bid pred"].iloc[idx]

        seq = {
                "past seq": past_seq,
                "Ask pred": Ask_pred,
                "Bid pred":Bid_pred
        }

        return seq



class LSTM_Model(nn.Module):
    def __init__(self, hidden_size, n_layers, batch_size, dropout_rate = 0.8, device = torch.device("cuda") ):
        super(LSTM_Model, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size = 2,
            hidden_size = hidden_size,
            batch_first = True,
            num_layers = n_layers,
            bidirectional = False
        )

        self.h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_size, dtype = torch.double).to(self.device)
        self.c_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_size, dtype = torch.double).to(self.device)


        self.dropout = nn.Dropout(p = dropout_rate)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.fc_p = nn.Linear(self.hidden_size, 1)
        self.soft = nn.Softplus()
        self.sig = nn.Sigmoid()


    def forward(self, batch):
        past_seq = batch["past seq"].to(self.device)
        out, hidden = self.lstm(past_seq, (self.h_0, self.c_0))
        out = out[:,-1,:]
        out = self.dropout(out)
        out_l = self.fc(out)
        out_l = self.soft(out_l)
        out_p = self.fc_p(out)
        out_p = self.sig(out_p)
        return out_l, out_p, batch["Ask pred"], batch["Bid pred"]


class WaveNet_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation, padding):
        super(WaveNet_layer, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel, dilation = dilation, padding = padding)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv1d(in_channels, in_channels+out_channels, 1)
        self.in_channels = in_channels

    def forward(self,x):
        y = self.conv(x)
        v = self.tanh(y)
        y = self.sig(y)

        out = self.conv1(y*v)

        y = out[:,:self.in_channels,:]
        v = out[:,self.in_channels:,:]

        return x+y, v

class CNN_Model(nn.Module):
    def __init__(self, out_channels, window_size, dropout_rate = 0.8, device = torch.device("cuda")):
        super(CNN_Model, self).__init__()
        self.device = device
        self.conv = nn.Conv1d(2, out_channels, 3, padding = 1)
        self.tanh = nn.Tanh()
        self.block = self._block(out_channels, out_channels, 4)
        self.conv1 = nn.Conv1d(out_channels, 1, 1)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_p = nn.Linear(window_size,1)
        self.fc_lmbd = nn.Linear(window_size,1)
        self.soft = nn.Softplus()
        self.sig = nn.Sigmoid()


    def _block(self, in_channels, out_channels, layers):
        models = []
        for i in range(layers):
            models.append(
                WaveNet_layer(in_channels, out_channels, kernel = 3, dilation=2**i, padding=2**i)
            )
        return nn.ModuleList(models)

    def forward(self, batch):
        seq = batch["past seq"].transpose(1,2).to(self.device)
        out = self.tanh(self.conv(seq))
        cum_sum = 0
        for i, conv in enumerate(self.block):
            out, v = conv(out)
            if (i ==0 ):
                cum_sum = v
            else:
                cum_sum = cum_sum + v
        out = self.conv1(self.relu(cum_sum))
        out = self.dropout(out)
        out = self.elu(out)
        out_fc_p = self.sig(self.fc_p(out))
        out_fc_lmbd = self.soft(self.fc_lmbd(out))
        return out_fc_lmbd, out_fc_p, batch["Ask pred"], batch["Bid pred"]


def poissonLoss(predicted, observed):
    loss=torch.mean(predicted-observed*torch.log(predicted))
    return loss

bceloss = nn.BCELoss();

def poissonLossPlus(lmbd, p, observed, clas, bce = bceloss):
    loss_1 = torch.mean(clas*(lmbd-observed*torch.log(lmbd)))
    loss_2 = bce(p, clas)
    return loss_1 + loss_2












