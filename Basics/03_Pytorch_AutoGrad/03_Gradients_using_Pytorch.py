# import numpy as np
import torch

# X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
# y = np.array([2, 4, 6, 8, 10], dtype=np.float32)
X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

# w = 0.1
w = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)


# ============== prediction ============== #
def forward(x):
    return w * x


# Prediction before Training
print(f'Prediction before Training: f(5)= {forward(5):.4f}')


# ============= loss --> MSE ============= #
def loss(y, y_pred):
    return ((y - y_pred) ** 2).mean()


# =============== gradient ============= #
# def gradient(X, y, y_pred):
#     return np.dot(2*X,y_pred - y).mean()


# Setting Hyper Parameters
lr = 0.01
n_iter = 50

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
    # w -= lr * dw
    with torch.no_grad():
        w -= lr * w.grad

    """
    Once weights are updated, we should make all of our gradients empty or zero (inplace) for
    next iteration because, whenever we call error.backward(), it will calculate the gradients
    and accumulate them into w.grad attribute.
    """
    w.grad.zero_()

    if epoch % 5 == 0:
        print(f'epoch = {epoch}: weights = {w:.3f}, loss = {error:.8f}')

# Prediction after Training
print(f'Prediction after Training: f(5)= {forward(5):.4f}')
