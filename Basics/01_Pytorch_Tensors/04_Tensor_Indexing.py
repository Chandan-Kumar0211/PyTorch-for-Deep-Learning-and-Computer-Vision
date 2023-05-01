import torch

# ---------------------------------------------------------------------------------- #
# ==============================---Tensor Indexing---=============================== #
# ---------------------------------------------------------------------------------- #

batch_size = 10
features = 25
x = torch.rand(batch_size, features)
# print(x)

# features of 1st data point
print(x[0])  # similar to x[0,:]

# 1st feature of all the data points:
print(x[:, 0])

# Third example in the batch and its first 10 features
print(x[2, :10])

# Assign 100 to 4th features of 5th data point
x[4, 3] = 100
print(x[4])


# ========Fancy Indexing======== #
f1 = torch.arange(10)
indices = [2, 5, 8]
print(f1[indices])


f2 = torch.rand((3,5))
rows = torch.tensor([1, 0])
col = torch.tensor([4, 2])
print(f2)
print(f2[rows, col])  # First it will print f2[1,4] and then f2[0,2]


# ==========Advance Indexing========= #
adv = torch.arange(10)
print(adv[(adv < 3) | (adv > 7)])   # Applying OR operation
print(adv[(adv < 3) & (adv > 7)])   # Applying AND operation
print(adv[adv.remainder(2) == 0])


# =======Some most useful operations======== #
y = torch.arange(10)

# re-evaluating the values of tensor using some conditions
print(torch.where(y > 5, y, y*2))  # if the condition y>5 is satisfied then it will print y else print y*2

# getting unique elements from a tensor
print(torch.tensor([0,1,1,2,4,4,4,8,1]).unique())

# Finding no of dimension of a tensor
print(y.ndimension())

# Finding total no of elements in a tensor
print(y.numel())