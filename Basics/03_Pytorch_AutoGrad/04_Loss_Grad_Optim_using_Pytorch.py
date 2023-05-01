import torch
from torch import nn

X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

w = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)


# ============== prediction ============== #
def forward(x):
    return w * x


# Prediction before Training
print(f'Prediction before Training: f(5)= {forward(5):.4f}')


# ============= loss --> MSE ============= #
# def loss(y, y_pred):
#     return ((y - y_pred) ** 2).mean()
loss = nn.MSELoss()


# =============== gradient ============= #
# def gradient(X, y, y_pred):
#     return np.dot(2*X,y_pred - y).mean()


# Setting Hyper Parameters
lr = 0.01
n_iter = 50


# ============= Optimizer ============== #
# There are various type ogf optimizers present,
# we can choose any one of them based on our requirements
optimizer = torch.optim.SGD([w], lr=lr)


# ================== Training =============== #
for epoch in range(n_iter):
    # Step:01- Prediction (forward pass)
    y_pred = forward(X)

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

    if epoch % 5 == 0:
        print(f'epoch = {epoch}: weights = {w:.3f}, loss = {error:.8f}')

# Prediction after Training
print(f'Prediction after Training: f(5)= {forward(5):.4f}')
