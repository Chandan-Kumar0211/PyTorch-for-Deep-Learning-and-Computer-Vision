import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):
    def __init__(self, transform=None):  # for data loading
        xy = np.loadtxt('../Dataset_Collection/wine.csv',
                        delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]  # NOTE: Here, we are not converting them into tensors
        self.y = xy[:,[0]]
        # self.x = torch.from_numpy(xy[:, 1:])   # Earlier we converted them into tensors
        # self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):  # getting data using indexing
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):  # length of dataset
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        features, labels = sample
        return torch.from_numpy(features), torch.from_numpy(labels)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self,sample):
        features, labels = sample
        features *= self.factor
        return features, labels


# ================== Applying Tensor Transform ==================== #
data_t1 = WineDataset(transform=ToTensor())
first_data_t1 = data_t1[0]
inputs_t1 , output_t1 = first_data_t1
print("\n",f'Features: {inputs_t1}')
print("\n",f'Type of Features: {type(inputs_t1)}, \n Type of Labels: {type(output_t1)}')


# ================== Applying Multiplication Transform ==================== #
data_t2 = WineDataset(transform=MulTransform(2))
first_data_t2 = data_t2[0]
inputs_t2, output_t2 = first_data_t2
print("\n",f'Features: {inputs_t2}')
print("\n",f'Type of Features: {type(inputs_t2)}, \n Type of Labels: {type(output_t2)}')


# ================== Applying Composed Transform ==================== #
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
data = WineDataset(transform=composed)
first_data = data[0]
inputs, output = first_data
print("\n",f'Features: {inputs}')
print("\n",f'Type of Features: {type(inputs)}, \n Type of Labels: {type(output)}')