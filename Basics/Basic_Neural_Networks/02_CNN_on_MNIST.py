import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms


# Creating Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self,in_channel, num_classes):
        super(CNN, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, 84)
        self.fc2 = nn.Linear(84, 32)
        self.fc3 = nn.Linear(32, num_classes)

        # self.nn_layers = nn.ModuleList()

    def forward(self, x):
        # Step:1- Feature Extraction
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)

        # Step:2- Reshaping/Flattening & Classification
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # We should have applied 'softmax' activation function on last layer
        # but, it will be taken cared by 'criterion' i.e., CrossEntropyLoss
        return out


# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# setting hyper-parameters
in_channel = 1
num_classes = 10
batch_size = 16
lr = 0.001
num_epochs = 3


# Loading Data
train_data = torchvision.datasets.MNIST(root="../Dataset_Collection/", download=True, train=True,
                                        transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = torchvision.datasets.MNIST(root="../Dataset_Collection/", download=True, train=False,
                                       transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# Initializing our model
model = CNN(in_channel=in_channel, num_classes=num_classes).to(device=device)


# Setting Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training our model
n_total_steps = len(train_loader)   # (total_num_training_sample) / (batch_size)
for epoch in range(num_epochs):
    for i, (data, actual_out) in enumerate(train_loader):
        data = data.to(device=device)
        actual_out = actual_out.to(device=device)

        # forward propagation
        pred_out = model.forward(data)
        loss = criterion(pred_out, actual_out)

        # backward propagation
        optimizer.zero_grad()   # for fresh gradients calculation in each batch
        loss.backward()

        # gradient descent
        optimizer.step()

        # debugging stuffs
        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}/{num_epochs}, steps: {i + 1}/{n_total_steps}, loss: {loss.item():.4f}')


# Function for checking accuracy of our model
def check_accuracy(loader, our_model):
    if loader.dataset.train:
        print("\nChecking accuracy on Training data:")
    else:
        print("\nChecking accuracy on Test data:")

    num_correct = 0
    num_samples = 0
    our_model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = our_model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy'
              f' {float(num_correct) / float(num_samples) * 100:.2f}')


# checking accuracy
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
