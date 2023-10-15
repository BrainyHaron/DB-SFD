import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from models import *

torch.set_printoptions(linewidth=1000)
drive_path = "."
train_y = np.load(drive_path+"/train/y_smp_train_scaled01.npy")
test_y = np.load(drive_path+"/test/y_smp_test_scaled01.npy")
train_pars = np.load(drive_path+"/train/pars_smp_train.npy")
train_pars = np.reshape(train_pars, train_pars.shape[:2])

#USE_CUDA = torch.cuda.is_available()
#device = 'cuda' if USE_CUDA else 'cpu'
device = 'cpu'

test_y = test_y.reshape(len(test_y),1,200,3)


model_dirs = {
    'mean       :': "./model_0",
    'q10        :': "./model_1",
    'q25        :': "./model_2",
    'q50        :': "./model_3",
    'q75        :': "./model_4",
    'q90        :': "./model_5"

}

check_idx = 2
example = torch.Tensor([train_y[check_idx]]).to(device)
models=[None]*6
with torch.no_grad():
    print("real answer:",torch.Tensor([train_pars[check_idx]]))
    for i, k in enumerate(model_dirs):
        models[i] = CNNRegression_3(n_hidden=100, n_outputs=15).to(device)
        models[i].load_state_dict(torch.load(model_dirs[k]))
        models[i].eval()
        print(k, models[i](example))
print(test_y.shape)
with torch.no_grad():
    result = np.zeros((len(test_y),15,6))
    for i in range(len(test_y)):
        if i % 10000==0:
            print(i)
            #np.save("./test/pars_test_{}.npy".format(i), result)
        for j in range(6):
            result[i,:,j] = models[j](torch.Tensor(test_y[i]).to(device)).to('cpu')
print(result.shape)
np.save("./test/pars_test.npy", result)
