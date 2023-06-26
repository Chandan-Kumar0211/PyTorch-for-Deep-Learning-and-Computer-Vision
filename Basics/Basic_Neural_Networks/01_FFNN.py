import torch
import torch.nn as nn  # contains all basics neural networks
import torch.optim as optim  # have all the optimization algorithm
import torch.nn.functional as F  # all the functions that don't have parameters e.g. relu, tanH
from torch.utils.data import DataLoader  # provide ease in dataset management
import torchvision.datasets  # contains many Dataset_Collection
import torchvision.transforms as transforms


#  Feed Forward Neural Network
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_layer, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


# Setting Device - CPU/GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting HyperParameters
input_size = 784  # 28*28 - MNIST image size after flattening
hidden_layer = 128
num_classes = 10  # all 10 digits
lr = 0.001
batch_size = 64
num_epochs = 5

# Downloading and Loading Data
train_data = torchvision.datasets.MNIST(root="../Dataset_Collection", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = torchvision.datasets.MNIST(root="../Dataset_Collection", train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Initializing our model
model = FFNN(input_size=input_size,hidden_layer=hidden_layer, num_classes=num_classes).to(device)

# Setting type of Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training our Model
n_total_steps = len(train_loader)   # (total_num_training_sample) / (batch_size)
for epoch in range(num_epochs):
    # for data, targets in train_loader:
    for i, (data, targets) in enumerate(train_loader):
        # Get data to CUDA if available
        data = data.to(device=device)  # Images
        targets = targets.to(device=device)  # Labels

        # flattening the image matrix to a vector while conserving the batch size
        # print(data.shape)  --> [64, 1, 28, 28]
        data = data.reshape(data.shape[0], -1)  # [64, 784]

        # forward propagation
        outputs = model(data)  # This is callable instance of class NN
        loss = criterion(outputs, targets)  # calculate loss btw outputs and actual labels/targets

        # back propagation
        optimizer.zero_grad()  # fresh new values for each batch
        loss.backward()
        optimizer.step()  # update the parameters (gradient descent)

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, steps: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')


# checking accuracy of the model
def check_accuracy(loader, our_model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    our_model.eval()  # so that pytorch knows that here we are not training but evaluating

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)  # images
            y = y.to(device=device)  # labels

            x = x.reshape(x.shape[0], -1)

            output = our_model(x)
            _, predictions = torch.max(output, 1)  # It will give --> value, index

            num_correct += (predictions == y).sum()
            num_samples += y.shape[0]

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
