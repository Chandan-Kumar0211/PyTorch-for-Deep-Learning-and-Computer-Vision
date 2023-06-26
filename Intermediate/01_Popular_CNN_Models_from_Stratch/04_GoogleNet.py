import torch
from torch import nn


# Helper class to build Inception Blocks and also GoogleNet Blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))



#                     /-> out_1x1                    -->\
#                    /-> downSam_3x3 --> out_3x3      -->\
# previous_layer -->---> downSam_5x5 --> out_5x5       ---> --> filter_concatenation
#                     \-> maxPool_3x3 --> outPool_1x1-->/
class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_1x1, downSam_3x3,
                 out_3x3, downSam_5x5, out_5x5, outPool_1x1):
        super(InceptionBlock, self).__init__()

        self.block_1 = ConvBlock(in_channel, out_1x1, kernel_size=1)

        self.block_2 = nn.Sequential(
            ConvBlock(in_channel, downSam_3x3, kernel_size=1),
            ConvBlock(downSam_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.block_3 = nn.Sequential(
            ConvBlock(in_channel, downSam_5x5, kernel_size=1),
            ConvBlock(downSam_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel, outPool_1x1, kernel_size=1)
        )

    def forward(self,x):
        # n_images_in_a_batch * num_filters * width * height
        # here, we need to concatenate all the filters ==> so we choose 1
        return torch.cat([self.block_1(x), self.block_2(x), self.block_3(x),self.block_4(x)], 1)


class GoogleNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = ConvBlock(in_channel, out_channel=64, kernel_size=(7,7),
                               stride=(2,2), padding=(3,3))
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.lrn = nn.LocalResponseNorm(2)

        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128,)
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.drop = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=1000)

    def forward(self, x):
        # feature extraction
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.lrn(out)   # Normalization used in GoogleNet Architecture
        # out = self.conv2(out)
        out = self.conv2(out)
        out = self.lrn(out)
        out = self.max_pool(out)

        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.max_pool(out)

        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4b(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.max_pool(out)

        out = self.inception_5a(out)
        out = self.inception_5b(out)
        out = self.avg_pool(out)

        out = self.drop(out)
        out = self.fc(out)
        return out

