
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader


from .Architechtures import OrderArrival_Dataset, CNN_Model, poissonLoss, poissonLossPlus


def train_one_epoch(model, optimizer, dataloader, which, batch_size, training = True, device = torch.device("cpu")):

    if (training):
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_pred = []
    total_p    = []
    true_labels = []
    for i, batch in enumerate(dataloader,1):
        if (training):
            optimizer.zero_grad()
        lmbd, p, y1, y2 = model(batch)
        y = y2
        if (which == "Ask"):
            y = y1
        clas = torch.tensor(1.0*(np.array(y.view(-1)) > 0.0))
        lmbd = lmbd.view(-1)
        p = p.view(-1)
        y = y.view(-1).to(device)
        loss = poissonLossPlus(lmbd,p, y, clas.to(device))
        
        if (training):
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_pred += lmbd.cpu().detach().tolist()
        total_p    += p.cpu().detach().tolist()
        true_labels+= y.cpu().detach().tolist()

    total_loss = total_loss/len(dataloader.dataset)*batch_size

    total_loss  = np.array(total_loss)
    total_pred  = np.array(total_pred)
    total_p     = np.array(total_p   )
    true_labels = np.array(true_labels)


    return (total_loss, total_pred, total_p, true_labels)


def train(model,optimizer, trainLoader, testLoader, which, batch_size, epochs = 10, if_test = False, 
          device = torch.device("cuda")):

    for epoch in range(epochs):

        if (epoch == 8 or epoch == 12):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train_loss, train_pred, train_p, train_labels = train_one_epoch(
                                                    model,
                                                    optimizer,
                                                    trainLoader,
                                                    which,
                                                    batch_size,
                                                    device)

        if(if_test):
            test_loss, test_pred, test_p, test_labels = train_one_epoch(
                                                    model,
                                                    optimizer,
                                                    testLoader,
                                                    which,
                                                    batch_size,
                                                    training = False,
                                                    device = device)


            print("epoch: %02d, Train Loss: %0.4f, Test Loss: %0.4f"
                 %(epoch, train_loss, test_loss))
        else:

            print("epoch: %02d, Train Loss: %0.4f"
                 %(epoch, train_loss))

def evaluate(models, data, window_size = 400, batch_size = 8, append = False):

    proper_data = data[["Bid Arrival", "Ask Arrival", "Bid Arr Pred", "Ask Arr Pred"]]
    proper_data.columns = ["Bid Arrival", "Ask Arrival", "Bid pred", "Ask pred"]
    
    dataset = OrderArrival_Dataset(proper_data, window_size)
    dataLoader = DataLoader(dataset, batch_size, shuffle = False, drop_last = False)
    
    bid_loss, bid_pred, bid_p, bid_labels = train_one_epoch(
                                    models[0], None, dataLoader, "Bid", batch_size, training = False)
    
    ask_loss, ask_pred, ask_p, ask_labels = train_one_epoch(
                                    models[1], None, dataLoader, "Ask", batch_size, training = False)
    
    
    from sklearn.metrics import f1_score, roc_auc_score
    
    auc_ask = roc_auc_score(ask_labels > 0.0, ask_p)
    auc_bid = roc_auc_score(bid_labels > 0.0, bid_p)
    
    f1_ask  = f1_score(ask_labels >0, ask_p > 0.5)
    f1_bid  = f1_score(bid_labels >0, bid_p > 0.5)
    
    ask_accu = np.mean( (ask_labels > 0) == (ask_p > 0.5) )
    bid_accu = np.mean( (bid_labels > 0) == (bid_p > 0.5) )
    
    res = np.array([[bid_loss, ask_loss],
                    [auc_bid , auc_ask ],
                    [f1_bid,   f1_ask  ],
    				[bid_accu, ask_accu]])
    resFrame = pd.DataFrame(res, columns=["bid_model", "ask_model"], index= ["loss", "AUC", "f1_score", "accuracy"])
    
    print (resFrame)
    
    
    start = 0
    
    data["DeepNet bid p"   ] = np.zeros(data.shape[0])
    data["DeepNet bid pred"] = np.zeros(data.shape[0])
    data["DeepNet ask p"   ] = np.zeros(data.shape[0])
    data["DeepNet ask pred"] = np.zeros(data.shape[0])

    for i, date in enumerate(dataset.dates):
        index = (data.index.floor("D") == date)
        bid_append = np.ones((np.sum(index),2)) * np.array([bid_p.mean(), bid_pred.mean()])
        ask_append = np.ones((np.sum(index),2)) * np.array([ask_p.mean(), ask_pred.mean()])

        bid_append[window_size-1:-1,0] = bid_p   [start:start + dataset.lengths[i]]   
        bid_append[window_size-1:-1,1] = bid_pred[start:start + dataset.lengths[i]]

        ask_append[window_size-1:-1,0] = ask_p   [start:start + dataset.lengths[i]]
        ask_append[window_size-1:-1,1] = ask_pred[start:start + dataset.lengths[i]]

        start += dataset.lengths[i]
        data["DeepNet ask p"   ][index] = ask_append[:,0]
        data["DeepNet ask pred"][index] = ask_append[:,1]
        data["DeepNet bid p"   ][index] = bid_append[:,0]
        data["DeepNet bid pred"][index] = bid_append[:,1]
    if (append):
        return data
    else :
        new_data = data[["DeepNet ask p", "DeepNet ask pred", "DeepNet bid p", "DeepNet bid pred"]]
        return new_data

def load_and_evaluate(paths, data, channel_num = 16,  window_size = 400, batch_size = 8, append = False):
   
    cpu = torch.device("cpu")
    bid_model = CNN_Model(channel_num, window_size, device=cpu)
    bid_model.load_state_dict(torch.load(paths[0], map_location= cpu) )
    bid_model.double()
    ask_model = CNN_Model(channel_num, window_size, device=cpu)
    ask_model.load_state_dict(torch.load(paths[1], map_location= cpu) )
    ask_model.double()
    return evaluate([bid_model, ask_model],data, window_size, batch_size, append)
    
if __name__=="__main__":


    device        = torch.device("cpu")
    
    batch_size    = 8
    hidden_size   = 64
    n_layers      = 2
    dropout_rate  = 0.8
    lr            = 0.001
    channel_num   = 32
    window_size   = 400
    
    
    
    data = pd.read_csv("../data.csv")
    data["Time"] = pd.to_datetime(data["Time"])
    data.set_index("Time", inplace = True)
    
    
    days = set(data.index.floor("D"))
    
    data["Bid Arr Pred"] = data["Bid Arrival"].shift(-1)
    data["Ask Arr Pred"] = data["Ask Arrival"].shift(-1)
    
    test_data = data.loc[data.index.floor("D") == pd.Timestamp('2017-06-01 00:00:00')]
    train_data = data.loc[data.index.floor("D") != pd.Timestamp('2017-06-01 00:00:00')]
    
    test_data = test_data[["Bid Arrival", "Ask Arrival"]]
    train_data = train_data[["Bid Arrival", "Ask Arrival"]]
    
    test_data["Bid pred"] = test_data["Bid Arrival"].shift(-1)
    test_data["Ask pred"] = test_data["Ask Arrival"].shift(-1)
    
    train_data["Bid pred"] = train_data["Bid Arrival"].shift(-1)
    train_data["Ask pred"] = train_data["Ask Arrival"].shift(-1)
    
    train_data = train_data.fillna(method="ffill")
    test_data = test_data.fillna(method="ffill")
    
    test_data = np.round(test_data)
    train_data = np.round(train_data)
    
    
    
    trainset = OrderArrival_Dataset(train_data, window_size)
    testset = OrderArrival_Dataset(test_data,window_size)
    
    trainLoader = DataLoader(trainset, batch_size = batch_size, shuffle = True, drop_last = True )
    testLoader = DataLoader(testset, batch_size = batch_size, drop_last = True)
    
    #ask_model = LSTM_Model(hidden_size = 64, n_layers = 2, batch_size =batch_size , dropout_rate = 0.8 )
    ask_model = CNN_Model(channel_num, window_size, device = device)
    ask_model.double().to(device)
    
    #bid_model = LSTM_Model(hidden_size, n_layers, batch_size =batch_size , dropout_rate = 0.8 )
    bid_model = CNN_Model(channel_num, window_size, device = device)
    bid_model.double().to(device)
    
    
    ask_optimizer = torch.optim.Adam(ask_model.parameters(),lr = lr)
    bid_optimizer = torch.optim.Adam(bid_model.parameters(),lr = lr)
    
    
    train(ask_model, ask_optimizer, trainLoader, testLoader, "Ask" , batch_size, 16, True, device = device)
    train(bid_model, bid_optimizer, trainLoader, testLoader, "Bid" , batch_size, 16, True, device = device)
    
    evaluate([ask_model,bid_model], test_data, window_size)
    
    torch.save(ask_model.state_dict(), "ask_CNN.pth")
    torch.save(bid_model.state_dict(), "bid_CNN.pth")
    
