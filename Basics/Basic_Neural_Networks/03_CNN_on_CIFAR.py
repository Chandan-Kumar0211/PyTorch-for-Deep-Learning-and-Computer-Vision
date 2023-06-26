import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Defining our CNN Architecture
class CNN_CIFAR(nn.Module):
    def __init__(self, num_in_channel=3, num_classes=10):
        super(CNN_CIFAR,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_in_channel, out_channels=40,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=20,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=10,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.pool_max = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.pool_avg = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(10*4*4, 120)   # inp = (num_out_channels)*(width)*(height)
        self.fc2 = nn.Linear(120,80)
        self.fc3 = nn.Linear(80,40)
        self.fc4 = nn.Linear(40,num_classes)

    def forward(self, inputs):
        # Step:1- Feature Extraction
        # input --> 3*32*32
        out = self.pool_max(F.relu(self.conv1(inputs)))     # 40*(16*16)
        out = self.pool_max(F.relu(self.conv2(out)))        # 20*(8*8)
        out = self.pool_avg(F.relu(self.conv3(out)))        # 10*(4*4)

        # Step:2- Reshaping/Flattening and Classification
        out = out.view(-1, 10*4*4)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)     # No activation function applied (because softmax is applied
        return out              # during loss calculation through Cross Entropy


# Setting hyper-parameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10


# Downloading Data and Loading it and setting class to train
# ========================= NOTE ======================== #
# dataset has PILImages of range [0,1]
# We transform them to tensors and normalize them btw [-1,1]
# ----------------------------------------------------------#
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = datasets.CIFAR10(root="../Dataset_Collection/", download=True,
                              train=True, transform=data_transform)
test_data = datasets.CIFAR10(root="../Dataset_Collection/", download=True,
                             train=False, transform=data_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'horse', 'frog', 'ship', 'truck')


# Model Initialization
model = CNN_CIFAR().to(device=device)


# Setting Loss and Optimizer for Model training
criterion = nn.CrossEntropyLoss()   # as it is multiclass classification (10 classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training the Model
num_total_steps = len(train_loader)     # (total_num_training_sample) / (batch_size)
for epoch in range(num_epochs):
    for idx, (features, labels) in enumerate(train_loader):
        # setting calculation on gpu (cuda)
        features = features.to(device=device)
        labels = labels.to(device=device)

        # forward propagation
        y_pred = model.forward(features)
        loss = criterion(y_pred, labels)

        # backward propagation
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()   # empty the optimizer for fresh gradient calculation of next batch

        # debugging stuff
        if (idx+1) % 300 == 0:
            print(f'epoch: {epoch}/{num_epochs}, step: {idx+1}/{num_total_steps}',
                  f'loss: {loss.item():.4f}')

print("\nModel Training Done")

with torch.no_grad():
    num_correct = 0     # for whole test data
    num_samples = 0
    num_class_correct = [0 for i in range(10)]  # for each class in test data
    num_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model.forward(images)

        # torch.max() --> value, index
        _, predicted = torch.max(outputs, 1)
        num_samples += labels.size(0)
        num_correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if pred == label:
                num_class_correct[label] += 1
            num_class_samples[label] += 1

    acc = 100.0 * num_correct/num_samples
    print(f'\nAccuracy of the Network = {acc:.4f}')

    for i in range(10):
        acc_class = 100.0 * num_class_correct[i]/num_class_samples[i]
        print(f'Accuracy of {classes[i]} = {acc_class}')
