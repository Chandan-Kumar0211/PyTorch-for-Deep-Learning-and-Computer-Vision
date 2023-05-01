import torch
from torch import nn


class NeuralNet04(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layer=10, hidden_size=128, choice='rRelu'):
        super().__init__()

        self.deep_nn = nn.ModuleList()
        self.act = nn.ModuleDict({
            'rRelu': nn.ReLU(),  # regular ReLU
            'pRelu': nn.PReLU()  # parametric ReLU
        })

        for i in range(num_hidden_layer):
            self.deep_nn.add_module(f'fc_{i+1}',nn.Linear(input_size, hidden_size))
            self.deep_nn.add_module(f'activation_{i+1}',self.act[choice])
            input_size = hidden_size
        self.deep_nn.add_module('out_layer', nn.Linear(hidden_size, output_size))

    def forward(self,inputs):
        hidden_state = []
        for layer in self.deep_nn:
            print(layer)
            out = layer(inputs)
            hidden_state.append(out)
            inputs = out
        return hidden_state[-1], hidden_state


x = torch.randn(5,12)
model = NeuralNet04(12, 3, choice='pRelu')  # advantage of using ModuleDict
output, states = model.forward(x)
print(output)