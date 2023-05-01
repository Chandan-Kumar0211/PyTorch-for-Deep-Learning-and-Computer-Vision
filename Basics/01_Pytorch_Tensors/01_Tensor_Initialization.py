import torch

# ##############################---Initializing Tensor---################################ #

my_tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(my_tensor1)

# We can set on which device this tensor should be on i.e., either CPU or Cuda-GPU
# Note: By default its set to CPU
# Also, we can set other params like requires_grad=True; which are useful for computing gradients

my_tensor2 = torch.tensor([[11, 2, 5], [14, 5, 8]], dtype=torch.float32,
                          device="cuda", requires_grad=True)
# print(my_tensor2)

# Below is the code format people often use to run the code on any device
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor3 = torch.tensor([[0, 12, 5], [4, 55, 7]], dtype=torch.float32,
                          device=device, requires_grad=True)
# print(my_tensor3)
# print(my_tensor3.dtype)
# print(my_tensor3.device)
# print(my_tensor3.shape)
# print(my_tensor3.requires_grad)

# Other common Initializing method

# 1) An empty tensor whose values are uninitialized (means they are simply memory no's)
emt_tensor = torch.empty(size=(3, 3))

# 2) Zero tensor with all values zeros
zeros_tensor = torch.zeros((4, 5))

# 3) Ones- A tensor with all values 1
ones_tensor = torch.ones((4, 4))

# 4) Eye- A tensor with diagonal values 1 and others 0
eye_tensor = torch.eye(5, 6)

# 5) rand - A tensor with all values filled randomly btw 0 and 1
rand_tensor = torch.rand((2, 3))

# 6) diagonal -
diag_tensor = torch.diag(torch.ones(3))

print("emt_tensor:\n",emt_tensor)
print("zeros_tensor:\n",zeros_tensor)
print("ones_tensor:\n",ones_tensor)
print("eye_tensor:\n",eye_tensor)
print("rand_tensor:\n",rand_tensor)
print("diag_tensor:\n",diag_tensor)


# ==========using arange function============ #
x = torch.arange(start=0, end=10, step=2)  # excluding 'end'
print(x)

# ===========using linspace function=========== #
y = torch.linspace(start=0, end=10, steps=5)  # including 'start' and 'end'
print(y)

# ===========Applying Normal Distribution========== #
n = torch.empty(size=(4, 5)).normal_(mean=0, std=1)
print(n)

# ==========Applying Uniform Distribution========= #
# It is similar to rand function but here we can give upper and lower values instead of 0 and 1
u = torch.empty(size=(4, 5)).uniform_(2, 5)
print(u)

