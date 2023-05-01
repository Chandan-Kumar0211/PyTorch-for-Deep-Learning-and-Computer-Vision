import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):  # for data loading
        xy = np.loadtxt('../Dataset_Collection/wine.csv',
                        delimiter=",",
                        dtype=np.float32,
                        skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):  # getting data using indexing
        return self.x[index], self.y[index]

    def __len__(self):  # length of dataset
        return self.n_samples


data = WineDataset()
# first_data = data[0]
# features, labels = first_data
# print(features, "\n", labels)

dataloader = DataLoader(dataset=data, batch_size=8, shuffle=True)
data_iter = iter(dataloader)
next_data = next(data_iter)
features, labels = next_data
print(features,"\n", labels)


# ================== Dummy Training Loop ================= #
num_epoch = 2
total_samples = len(data)
num_iter = math.ceil(total_samples/8)  # rounds a number UP to the nearest integer
print("\n",total_samples, num_iter,"\n")

for epoch in range(num_epoch):
    for idx, (inputs, target) in enumerate(dataloader):
        if idx % 10 == 0:
            print(f'epoch: {epoch}/{num_epoch}, step: {idx + 1}/{num_iter}, inputs: {inputs.shape}')
