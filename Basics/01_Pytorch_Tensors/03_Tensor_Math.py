import torch

# ---------------------------------------------------------------------------------- #
# ================================---Tensor Maths---================================ #
# ---------------------------------------------------------------------------------- #
x = torch.tensor([1, 2, 3, 5])
y = torch.tensor([4, 3, 4, 2])
z = torch.tensor([2])
print("x: ", x)
print("y: ", y)
print("z: ", z)

# =========ADDITION============= #
a1 = torch.empty(4)  # Method-01
torch.add(x, y, out=a1)
print("a1: ", a1)

a2 = torch.add(x, y)  # Method-02
print("a2: ", a2)

a3 = x + y  # Method-03
print("a3: ", a3)

# ===========SUBTRACTION=========== #
s = x - y
print("s: ", s)

# ============DIVISION============= #
# dividing with same shape
d1 = torch.true_divide(x, y)
print("d1: ", d1)

# dividing with diff. shape (a tensor with an integer (scaler tensor))
d2 = torch.true_divide(x, z)
print("d1: ", d2)

# =======Inplace Operation========== #
t = torch.zeros(4)
print("t: ", t)

t.add_(x)
"""
wherever there is any operation followed by an underscore; it implies that the operation is done inplace
NOTE: Here; t.add_(x) is equal to "t += x" but not equal to "t = t + x"
"""

print("Inplace Operation on t: ", t)

# =======Exponentiation============= #
r = torch.rand(3, 3)
exp_r = r.pow(2)  # Or exp_x = x**2
print("Exponent of each element of r:\n", exp_r)

# ========Simple Comparison========= #
comp = x > 2
print("comp:\n", comp)

# ========Matrix Multiplication========== #
m1 = torch.rand(3, 5)
m2 = torch.rand(5, 4)
matrix_mul = torch.mm(m1, m2)  # Output size will of (3,4)
# Or: matrix_mul = m1.mm(m2)
print("matrix_mul:\n", matrix_mul)

# ========Matrix Exponentiation========== #
m3 = torch.rand((5, 5))
matrix_exp = m3.matrix_power(3)
print("matrix_exp:\n", matrix_exp)

# =========Element wise Multiplication========== #
ewm = x * y
print("ewm:\n", ewm)

# ==========Dot Product========== #
dp = torch.dot(x, y)
print("dp:\n", dp)  # it performs element wise multiplication followed by adding them together


# ===============Batch Matrix Multiplication============== #
batch = 32
n = 10
m = 23
p = 30

t1 = torch.rand(batch, n, m)
t2 = torch.rand(batch, m, p)
out_bmm = torch.bmm(t1, t2)  # Output size --> (n, p)
print("Batch Matrix Multiplication \n", out_bmm)


# ========Example of Broadcasting========= #
b1 = torch.rand(4, 4)
b2 = torch.rand(1, 4)

z1 = b1 - b2
z2 = b1 ** b2
# The above operations doesn't make sense in mathematics
# But in pytorch; the operation take place by expanding the no of similar rows until it matches the dimension


# ============Other useful Tensor Operations============== #

sum_x = torch.sum(x, dim=0)  # Or x.sum(dim=0)

val_max, indices_max = torch.max(x, dim=0)  # Or x.max(dim=0)
val_min, indices_min = torch.min(x, dim=0)  # Or x.min(dim=0)

only_ind_min = torch.argmin(x, dim=0)
only_ind_max = torch.argmax(x, dim=0)

abs_x = torch.abs(x)

mean_x = torch.mean(x.float(), dim=0)  # Conversion in float type is necessary
equal_or_not_equal = torch.eq(x, y)   # Element wise operation (boolean output)

sorted_y, indices_y_sorted = torch.sort(y, dim=0, descending=False)

# Used in ReLU
relu_x_min = torch.clamp(x, min=0)  # Set/Clamp all the values less than 0 to 0
relu_x_max = torch.clamp(x, max=2)  # Set/Clamp all the values greater than 2 to 2
relu_x_minmax = torch.clamp(x, min=2, max=3)
print(relu_x_min, relu_x_max, relu_x_minmax)

