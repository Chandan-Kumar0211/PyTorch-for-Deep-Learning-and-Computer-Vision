import torch

# =========================================================================================== #
# ---------------------------------------- NOTE --------------------------------------------- #

# We set the flag "required_grad = True", only in the argument of tensor with respect to which
# gradient need to be taken.
# =========================================================================================== #
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)
# Output --> tensor([1.7455, 3.5760, 1.7998], grad_fn=<AddBackward0>)

z = 2*y*y
print(z)
# Output --> tensor([ 6.0934, 25.5750,  6.4785], grad_fn=<MulBackward0>)

z = z.mean()
print(z)
# Output --> tensor(12.7292, grad_fn=<MeanBackward0>)


# ===================== Backward Propagation ============================= #
z.backward()
print(x.grad)
# Output --> tensor([2.7425, 3.9099, 3.8965])

