import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)
w = 1.0

# ============== prediction ============== #
def forward(x):
    return w*x

# Prediction before Training
print("\n",f'Prediction before Training: f(5)- {forward(5):.4f}',"\n")

# ============= loss --> MSE ============= #
# Since our loss(MSE): J = (1/N)*((w*x - y)**2)
def loss(y, y_pred):
    return ((y-y_pred)**2).mean()

# =============== gradient ============= #
# gradient of loss w.r.t weights : dJ/dw = (1/N)*(2*(w*x - y))*(2x)
def gradient(X, y, y_pred):
    return np.dot(2*X,y_pred - y).mean()


# Setting Hyper Parameters
lr = 0.001
n_iter = 100

# ================== Training =============== #
for epoch in range(n_iter):
    # Step:01- Prediction
    y_predicted = forward(X)

    # Step:02- Calculating Loss
    error = loss(y,y_predicted)

    # Step:03- Calculating Gradients
    dw = gradient(X,y,y_predicted)

    # Step:04- Updating weights
    w -= lr * dw

    if epoch % 10 == 0:
        print(f'epoch = {epoch}: weights = {w:.3f}, loss = {error:.8f}')


# Prediction after Training
print("\n",f'Prediction after Training: f(5)- {forward(5):.4f}',"\n")


