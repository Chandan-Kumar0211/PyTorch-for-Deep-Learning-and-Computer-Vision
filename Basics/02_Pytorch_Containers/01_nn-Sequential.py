import torch
from torch import nn

# ================================================================================================= #
# NOTE: Instead of using NeuralNet01 class (most basic method), we can use NeuralNet02 (sequential)
#       class which will be more organised, manageable and time saving.
# ================================================================================================= #


class NeuralNet01(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet01,self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.Sigmoid()
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.activation3 = nn.Softmax()

    def forward(self, inputs):
        out = self.layer1(inputs)
        out = self.activation1(out)
        out = self.layer2(out)
        out = self.activation2(out)
        out = self.layer3(out)
        return self.activation3(out)


class NeuralNet02(nn.Module):
    def __init__(self, in_size, out_size, num_hidden_layer=10, hidden_layer_size=128):
        super(NeuralNet02,self).__init__()

        # In PyTorch, the terms "module" and "layer" are often used interchangeably to refer to a
        # building block of a neural network that performs a specific computation on the input data.

        self.layers = nn.Sequential()  # initialization Sequential class

        for i in range(num_hidden_layer):
            self.layers.add_module(f'fc{i}', nn.Linear(in_size, hidden_layer_size))   # adding layers/modules
            self.layers.add_module(f'activation{i}', nn.ReLU())
            in_size = hidden_layer_size
        self.layers.add_module('classifier', nn.Linear(hidden_layer_size, out_size))

    def forward(self,inputs):
        out = self.layers(inputs)
        #  the layers in a Sequential are connected in a cascading way.
        # Means when we pass an input; the Sequential automatically passes/applies to each of the modules
        # it stores (which are each a registered submodule of the Sequential).
        return out


x = torch.randn(8,14)
model = NeuralNet02(14,6)
output = model.forward(x)
print(output)






