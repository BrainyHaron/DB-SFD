import torch.nn as nn
import torch


class CNNRegression_2(nn.Module):#архитектуру взял по вдохновению(случайно)

  def __init__(self, n_hidden, n_outputs, dropout=0.2):
    '''
    n_hidden: количество нейронов на скрытом слое
    n_outputs: выходной размер
    dropout: вероятность дропаута
    '''
    super().__init__()
    self.cnn = nn.Sequential(
      nn.Conv1d(in_channels=200, out_channels=150, kernel_size=1, stride=1, padding=1),
      nn.BatchNorm1d(150),
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(dropout),
      nn.Conv1d(in_channels=150, out_channels=100, kernel_size=2, stride=1, padding=1),
      nn.BatchNorm1d(100),
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Flatten()
    )
    self.relu=nn.ReLU()
    self.f2 = nn.Linear(100, n_hidden)
    self.f3 = nn.Linear(n_hidden, n_outputs)
  def forward(self, x):
    out = self.cnn(x)
    #out=self.drop(out)
    out=self.f2(out)
    out=self.relu(out)
    #out=self.drop(out)
    out=self.f3(out)
    return out


class CNNRegression_3(nn.Module):#архитектуру взял по вдохновению(случайно)

  def __init__(self, n_hidden, n_outputs, dropout=0.2):
    '''
    n_hidden: количество нейронов на скрытом слое
    n_outputs: выходной размер
    dropout: вероятность дропаута
    '''
    super().__init__()
    self.cnn = nn.Sequential(
      nn.Conv1d(in_channels=200, out_channels=170, kernel_size=3, stride=1, padding=2),
      nn.BatchNorm1d(170),
      nn.ReLU(),
      nn.MaxPool1d(2, stride=1),
      nn.Conv1d(in_channels=170, out_channels=140, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm1d(140),
      nn.ReLU(),
      nn.MaxPool1d(2, stride=1),
      nn.Dropout(dropout),
      nn.Conv1d(in_channels=140, out_channels=110, kernel_size=3, stride=1, padding=2),
      nn.BatchNorm1d(110),
      nn.ReLU(),
      nn.MaxPool1d(2, stride=1),
      nn.Conv1d(in_channels=110, out_channels=80, kernel_size=3),
      nn.BatchNorm1d(80),
      nn.ReLU(),
      nn.MaxPool1d(2, stride=1),
      nn.Dropout(dropout),#убрать для первой версии
      nn.Flatten()
    )
    self.relu=nn.ReLU()
    self.f2 = nn.Linear(80, n_hidden)
    self.f3 = nn.Linear(n_hidden, n_outputs)
  def forward(self, x):
    out = self.cnn(x)
    out=self.f2(out)
    out=self.relu(out)
    out=self.f3(out)
    return out


class CNNRegression_quantile(nn.Module):

  def __init__(self, n_hidden, n_outputs, dropout=0.2):
    '''
    n_hidden: количество нейронов на скрытом слое
    n_outputs: выходной размер
    dropout: вероятность дропаута
    '''
    super().__init__()
    self.cnn = nn.Sequential(
      nn.Conv1d(in_channels=200, out_channels=150, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm1d(150),
      nn.ReLU(),
      nn.MaxPool1d(2, stride=1),
      nn.Conv1d(in_channels=150, out_channels=100, kernel_size=2, stride=1, padding=1),
      nn.BatchNorm1d(100),
      nn.ReLU(),
      nn.MaxPool1d(2, stride=1),
      nn.Dropout(dropout),
      nn.Conv1d(in_channels=100, out_channels=50, kernel_size=2, stride=1, padding=1),
      nn.BatchNorm1d(50),
      nn.ReLU(),
      nn.MaxPool1d(2, stride=1),
      nn.Conv1d(in_channels=50, out_channels=25, kernel_size=2),
      nn.BatchNorm1d(25),
      nn.ReLU(),
      nn.Flatten()
    )
    self.relu=nn.ReLU()
    self.f2 = nn.Linear(25, n_hidden)
    self.f3 = nn.Linear(n_hidden, n_outputs)
  def forward(self, x):
    out = self.cnn(x)
    out=self.f2(out)
    out=self.relu(out)
    out=self.f3(out)
    return out