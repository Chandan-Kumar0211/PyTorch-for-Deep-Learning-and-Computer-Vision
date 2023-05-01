import torch
import numpy as np

# ##############################---Tensor types & Conversions---##################################### #

# Convert tensor to other types
tensor = torch.arange(5)
print(tensor)             # by-default int64

print(tensor.bool())      # boolean - True/False
print(tensor.short())     # int16
print(tensor.long())      # int64 (important)
print(tensor.half())      # float16
print(tensor.float())     # float32 (important)
print(tensor.double())    # float64


# Array to Tensor and Vise-Versa
np_array = np.array([2,15,4,0,4])          # an array
print(np_array)
print(np_array.dtype)

arr_tensor = torch.from_numpy(np_array)    # array to tensor
print(arr_tensor)

np_array_back = arr_tensor.numpy()         # tensor to array
print(np_array_back)
print(np_array_back.dtype)

