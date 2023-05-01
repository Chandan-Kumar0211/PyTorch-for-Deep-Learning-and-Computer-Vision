import torch

# ---------------------------------------------------------------------------------- #
# ==============================---Tensor Reshaping---============================== #
# ---------------------------------------------------------------------------------- #

x = torch.arange(9)
y = torch.arange(12)

"""
NOTE: There are mainly two methods of reshaping a matrix
1) view()
2) reshape()

--> view acts on contiguous tensors means tensors are stored contiguously in the memory 
--> In reshape, the position of tensors doesn't matter 
--> using reshape is a safer choice as if the tensors are not contiguous then it makes copies
--> But reshape on non-contiguous tensor means there will be performance loss
"""

x_3x3 = x.view(3,3)  # It will be contiguous
print(x_3x3)

y_4x3 = y.reshape(4,3)   # It will be also contiguous
print(y_4x3)

z1 = x_3x3.t()  # This is non-contiguous
# print(z1.view(9))  ---> This throws an error:
# RuntimeError: view size is not compatible with input tensor's size and
# stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...)

# There are two methods for not getting error:
# 1) Simply using reshape() method
print(z1.reshape(9))
# 2) First making it contiguous and applying view() method
z2 = y_4x3.t()  # taking another example
print(z2.contiguous().view(12))


# Concatenation
t1 = torch.rand((2,5))
t2 = torch.rand((2,5))
print(torch.cat((t1,t2),dim=0))  # will add more rows
print(torch.cat((t1,t2),dim=0).shape)
print(torch.cat((t1,t2),dim=1))  # will add more columns
print(torch.cat((t1,t2),dim=1).shape)

# Flattening
t1_flat = t1.view(-1)   # Here, -1 means flattening

# reshaping only a few dimension
batch = 32
t = torch.rand((batch, 2, 5))
t_reshape = t.view(batch, -1)  # Output dim: [batch, 10]
print(t_reshape.shape)

# permute() function  (Note: Transpose is a special case of permute)
p = torch.arange(60)
p = p.view(3,5,4)
p = p.permute(0,2,1)

# Squeeze and Un-squeeze
s = torch.arange(10)
print(s.unsqueeze(0).shape)
print(s.unsqueeze(1).shape)

s1 = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(s1.shape)

s2 = s1.squeeze(0)
print(s2.shape)