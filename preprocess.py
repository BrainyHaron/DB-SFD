from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler()
drive_path = "."
train_y = np.load(drive_path+"/train/y_smp_train_scaled.npy")
test_y = np.load(drive_path+"/test/y_smp_test_scaled.npy")
for data in [train_y, test_y]:
  for x in data:
    scalers = [None]*3
    for i in range(3):
      scalers[i] = MinMaxScaler().fit(x[:,i].reshape(-1, 1))
    for i in range(3):
      x[:,i] = scalers[i].transform(x[:,i].reshape(-1, 1)).reshape(-1)
np.save(drive_path+"/train/y_smp_train_scaled.npy",train_y)
np.save(drive_path+"/test/y_smp_test_scaled.npy",test_y)
print(train_y)