import torch
from torch import nn, optim
from torch.nn import functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class LeNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=6,
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.pool_avg = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # feature extraction
        out = self.pool_avg(F.tanh(self.conv1(x)))
        out = self.pool_avg(F.tanh(self.conv2(out)))
        out = F.tanh(self.conv3(out))

        # reshaping --> 120*1*1 to 120
        out = out.reshape(out.shape[0], -1)

        # classification
        out = F.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


"""
# demo_input = torch.rand(64,1,32,32)
# demo_model = LeNet()
# print(demo_model(demo_input).shape)
"""

# configuring our device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initializing our model on gpu (as we have cuda)
model = LeNet().to(device=device)

# Setting hyper parameters
batch_size = 32
learning_rate = 0.01
num_epochs = 3

# defining loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# loading dataset
image_size = 32
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
train_data = datasets.MNIST(root="../../Basics/Dataset_Collection", download=False,
                            train=True, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root="../../Basics/Dataset_Collection", download=False,
                           train=False, transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Training
n_total_steps = len(train_loader)  # (total_num_training_sample) / (batch_size)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # computing on gpu
        images = images.to(device=device)
        labels = labels.to(device=device)

        # Calculating loss (forward pass)
        y_pred = model.forward(images)
        loss = criterion(y_pred, labels)

        # Calculating gradients and updating parameters (backward pass)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # debugging
        if i % 300 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, steps: {i}/{n_total_steps},'
                  f'loss: {loss.item():.4f}')

print("\nModel Training Done!!!\n")


def check_accuracy(loader, our_model):
    if loader.dataset.train:
        print("Checking accuracy for training set")
    else:
        print("Checking accuracy for test set")

    # letting the model know it's evaluating not training
    our_model.eval()

    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = our_model.forward(x)
            _, predictions = torch.max(scores, 1)

            num_correct += (predictions == y).sum()
            num_samples += (predictions.shape[0])

        print(f'Got {num_correct}/{num_samples} with accuracy'
              f' {float(num_correct) / float(num_samples) * 100:.2f}')


# accuracy
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


