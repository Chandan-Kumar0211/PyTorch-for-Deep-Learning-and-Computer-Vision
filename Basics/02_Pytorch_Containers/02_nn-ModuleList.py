import torch
from torch import nn


# ================================================================================================= #
# NOTE: If we want to access output of hidden layer, we won't be able to do it by using Sequential
#       container; so we need moduleList.
#       nn.ModuleList is used to store nn.Module instances,
#       nn.ModuleList is typically used to organize the modules of a neural network,
# ================================================================================================= #

class NeuralNet03(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layer=10, hidden_size=84):
        super(NeuralNet03, self).__init__()

        self.deep_nn = nn.ModuleList()

        for i in range(num_hidden_layer):
            self.deep_nn.add_module(f'fc{i}',nn.Linear(input_size, hidden_size))
            self.deep_nn.add_module(f'activation{i}', nn.ReLU())
            input_size = hidden_size
        self.deep_nn.add_module('Classifier', nn.Linear(hidden_size, output_size))

    def forward(self,inputs):
        hidden_state = []
        for layer in self.deep_nn:
            print(layer)
            out = layer(inputs)
            hidden_state.append(out)
            inputs = out
        return hidden_state[-1], hidden_state


x = torch.randn(5,12)
model = NeuralNet03(12, 3)
output, states = model.forward(x)
print(output)

