import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ============================ NOTE ============================ #
# Here, we are using MNIST dataset for training RNN model
# As we know, the shape in which we feed inputs to the models is:
# (N,1,28,28) --> here, N is batch size
# Earlier 28,28 was size of image but this time we will assume that;
# we are feeding the data 28 times in a sequence and each data of
# that sequence is having 28 features.
# Also note, 1 need to be removed from the input
# In general, it follows:- (N * sequence * features)
# ---------------------------------------------------------------------- #


# hyper-parameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.005
batch_size = 128
num_epochs = 3


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Building RNN Network
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Now taking all the output from hidden state to fc layer
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        # initializing hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagation
        out, _ = self.rnn(x, h0)

        # reshaping
        out = out.reshape(out.shape[0], -1)

        # Passing to fully connected at last
        out = self.fc(out)
        return out


# Downloading and Loading Data
train_data = datasets.MNIST(root="../Dataset_Collection", train=True, download=True,
                            transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root="../Dataset_Collection", train=False, download=False,
                           transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# model initialization
rnn = RNN(input_size=input_size, hidden_size=hidden_size,
          num_layers=num_layers, num_classes=num_classes).to(device)

# Defining Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

# Training the model
n_total_steps = len(train_loader)   # (total_num_training_sample) / (batch_size)
for epoch in range(num_epochs):
    for i, (features, targets) in enumerate(train_loader):
        # pushing calculations on GPU if cuda is available
        features = features.to(device).squeeze(1)   # (N,1,28,28) --> (N,28,28)
        targets = targets.to(device)

        # forward pass and loss calculation
        y_pred = rnn.forward(features)
        loss = criterion(y_pred, targets)

        # backward prop and weight updating using optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 300 == 0:
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
            x = x.to(device=device).squeeze(1)  # images
            y = y.to(device=device)  # labels

            output = our_model(x)
            _, predictions = torch.max(output, 1)  # It will give --> value, index

            num_correct += (predictions == y).sum()
            num_samples += y.shape[0]

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


check_accuracy(train_loader, rnn)
check_accuracy(test_loader, rnn)




