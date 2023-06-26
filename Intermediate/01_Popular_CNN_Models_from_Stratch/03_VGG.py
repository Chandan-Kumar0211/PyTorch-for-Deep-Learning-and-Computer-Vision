import torch
from torch import nn

"""
NOTE: --> Input image size of VGG network is: (3, 224, 224)
      --> VGG16 has a total of 16 layers that has some weights.
      --> Only Convolution and pooling layers are used.
      --> Always uses a 3 x 3 Kernel for convolution.
      --> 22×2 size of the max pool.
      --> 138 million parameters.
      --> Trained on ImageNet data.
"""

vgg_16 = [64,64,"max_pool",128,128,"max_pool",256,256,256,"max_pool",
          512,512,512,"max_pool",512,512,512,"max_pool",]

class VGG(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = input_channels
        self.conv_layers = self._feature_extractor(vgg_16)
        self.fc_layers = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self,x):
        # feature extraction
        out = self.conv_layers(x)

        # reshaping
        out = out.reshape(out.shape[0],-1)

        # classification
        out = self.fc_layers(out)

        return out

    # helper function for building all convolution network along with max pooling
    def _feature_extractor(self, network):
        layers = []
        in_channels = self.in_channels

        for n in network:
            if type(n) == int:
                out_channels = n
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3,3), stride=1, padding=1),
                           nn.ReLU()]
                in_channels = out_channels
            elif type(n) == str and n == "max_pool":
                # and operator is used if, in case; we have avg pool (in any modified version)
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)




demo_input = torch.rand(64,3,224,224)
demo_model = VGG()
print(demo_model(demo_input).shape)

# Output --> torch.Size([64, 1000])
