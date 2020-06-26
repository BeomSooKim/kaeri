from torch.utils.data import Dataset
import torch.nn as nn
import torch

class dfDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.target = y
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    
def weights_init(m, initializer = nn.init.kaiming_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        initializer(m.weight)
        

def train_model(model, train_data, weight, optimizer, loss_func):
    loss_sum = 0
    for i, (x, y) in enumerate(train_data):
        optimizer.zero_grad()
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    
    return loss_sum / len(train_data)

def eval_model(model, val_data, loss_func):
    with torch.no_grad():
        loss = 0
        for i, (x, y) in enumerate(val_data):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss += loss_func(pred, y).item()
    return loss / len(val_data)