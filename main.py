import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from models import *
drive_path = "."
train_y = np.load(drive_path+"/train/y_smp_train_scaled01.npy")
test_y = np.load(drive_path+"/test/y_smp_test_scaled01.npy")
train_pars = np.load(drive_path+"/train/pars_smp_train.npy")
train_pars = np.reshape(train_pars, train_pars.shape[:2])
#print(int(train_y.shape[0]/4))
#np.save(drive_path+"/train/y_smp_train_scaled_25percent.npy",train_y[:int(train_y.shape[0]/4)])
import matplotlib.pyplot as plt


class SochiDataset(Dataset):

    def __init__(self, y, par):#, column):
        self.y = y
        self.par = par
        #self.column=column

    def __getitem__(self, idx):
        return torch.Tensor(self.y[idx]), torch.Tensor(self.par[idx])

    def __len__(self):
        return len(self.y)


class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        self.q = quantile
        super(QuantileLoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.mean(torch.max((self.q - 1) * (targets - inputs), self.q * (targets - inputs)))


USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
BATCH_SIZE = 32 # Размер батча
split = 0.8 # Процент обучающих данных
#quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
criterions = [
    nn.MSELoss().to(device),
    QuantileLoss(0.1).to(device),
    QuantileLoss(0.25).to(device),
    QuantileLoss(0.5).to(device),
    QuantileLoss(0.75).to(device),
    QuantileLoss(0.9).to(device)
]
for par in range(len(criterions)):
    print("--------------------PARAMETER:{}-------------------------".format(par))
    dataset = SochiDataset(train_y, train_pars)
    #Разбиение на обучающую и валидационную выборки
    train_len = int(len(dataset)*split)
    lens = [train_len, len(dataset)-train_len]
    train_ds, test_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    lr = 0.001
    n_epochs = 10
    model = CNNRegression_3(n_hidden=100, n_outputs=15).to(device)
    criterion = criterions[par]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    t_losses, v_losses = [], []
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0
        model.train()
        with tqdm(trainloader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                train_loss += loss.item()
            epoch_loss = train_loss / len(trainloader)
            t_losses.append(epoch_loss)

        # validation step
        model.eval()
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                preds = model(x)
                error = criterion(preds, y)
            valid_loss += error.item()
        valid_loss = valid_loss / len(testloader)
        v_losses.append(valid_loss)

        print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
    torch.save(model.state_dict(), drive_path + '/model_{}'.format(par))