import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


# Step 00: Preparing data and Preprocessing
X_numpy, y_numpy = datasets.make_regression(n_samples=200, n_features=1, noise=20, random_state=44)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0],1)
n_samples, n_features = X.shape


# Step 01: Building a Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)


# Step 02: Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Step 03: Training the Model
num_epoch = 100
for epoch in range(num_epoch):
    # forward propagation
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward propagation
    loss.backward()

    # update parameters
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plotting
prediction = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy,'.')
plt.plot(X_numpy, prediction, 'g')
plt.show()