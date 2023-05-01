import torch
from torch import nn

# X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
# y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)
"""
NOTE: This time shape of our 'X' and 'y' will change because we need to feed them
      into Pytorch in-build model/models where, 
         # no of rows will represent --> no of samples
         # no of columns will represent --> no of features
"""
X = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

"""
We also don't require to manually initialize weights anymore, because
PyTorch model knows these and saves into model.parameters()
"""
# w = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)


# ============== prediction ============== #
# def forward(x):
#     return w * x
model = nn.Linear(in_features=input_size, out_features=output_size)


# Prediction before Training
# print(f'Prediction before Training: f(5)= {forward(5):.4f}')
print(f'Prediction before Training: f(5)= {model(X_test).item():.4f}')


# ============= loss --> MSE ============= #
# def loss(y, y_pred):
#     return ((y - y_pred) ** 2).mean()
loss = nn.MSELoss()


# =============== gradient ============= #
# def gradient(X, y, y_pred):
#     return np.dot(2*X,y_pred - y).mean()


# Setting Hyper Parameters
lr = 0.01
n_iter = 100

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# ================== Training =============== #
for epoch in range(n_iter):
    # Step:01- Prediction (forward pass)
    y_pred = model(X)

    # Step:02- Calculating Loss
    error = loss(y, y_pred)

    # Step:03- Calculating Gradients (backward pass)
    # dw = gradient(X, y, y_pred)
    error.backward()  # This will automatically calculate d(error)/dw

    # Step:04- Updating weights
    # with torch.no_grad():  # --> previous one
    #     w -= lr * w.grad
    optimizer.step()         # --> using pytorch

    """
    Once weights are updated, we should make all of our gradients empty or zero (inplace) for
    next iteration because, whenever we call error.backward(), it will write(calculate) our gradients
    and accumulate them into w.grad attribute.
    """
    # w.grad.zero_()        # --> previous one
    optimizer.zero_grad()   # --> using pytorch

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch = {epoch}: weights = {w[0][0]:.3f}, loss = {error:.8f}')

# Prediction after Training
# print(f'Prediction after Training: f(5)= {forward(5):.4f}')
print(f'Prediction before Training: f(5)= {model(X_test).item():.4f}')