import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets


# ============== Data preparation and preprocessing =================== #
bc_data = datasets.load_breast_cancer()
X, y = bc_data.data, bc_data.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# Scaling our features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Converting our data into tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)


# =========================== Model Building ============================== #
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression,self).__init__()
        self.lin = nn.Linear(input_dim, 1)

    def forward(self, x):
        pred = torch.sigmoid(self.lin(x))
        return pred


model = LogisticRegression(n_features)


# ======================= Loss and Optimizer ========================= #
learning_rate = 0.01

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# ======================== Training the Model ======================== #
num_epoch = 101

for epoch in range(num_epoch):

    # forward propagation
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)

    # backward propagation
    loss.backward()

    # updating weights
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss = {loss.item():.4f}')


# Evaluating our model
with torch.no_grad():
    y_prediction = model.forward(X_test)
    y_pred_cls = y_prediction.round()

    accuracy = y_pred_cls.eq(y_test).sum()/float(y_test.shape[0])
    print("accuracy", accuracy)

