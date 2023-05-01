import torch
from torch import nn
from torch.nn import functional as F


# ================================================================================================= #
# NOTE: nn.ParameterList is used to store nn.Parameter instances.
#       nn.ParameterList is typically used to organize the learnable parameters of a neural network.
# ================================================================================================= #

class NeuralNet05(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layer=10, hidden_size=128):
        super(NeuralNet05,self).__init__()
        
        self.deep_nn = nn.ParameterList()
        self.activation = nn.ReLU()

        for i in range(num_hidden_layer):
            self.deep_nn.append(nn.Parameter(torch.rand(hidden_size, input_size)))
            # above we are initializing the weights randomly
            input_size = hidden_size
        self.deep_nn.append(nn.Parameter(torch.rand(output_size, hidden_size)))

    def forward(self,inputs):
        hidden_states = []
        for idx, layer in enumerate(self.deep_nn):
            out = F.linear(inputs, layer)    # y = x.W^T + b;
            print(f'We are currently in layer {idx+1}')
            if idx != len(self.deep_nn)-1:   # will apply activation for all layer except last layer
                out = self.activation(out)
                print(f'--> The activation ran on layer {idx+1}\n')
            hidden_states.append(out)
            inputs = out
        return hidden_states[-1], hidden_states


x = torch.randn(4,12)
model = NeuralNet05(12,3)
output, states = model.forward(x)
print('\n',output)

