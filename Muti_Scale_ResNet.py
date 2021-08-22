import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel):
        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.conv1_1_1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=(7, 1), stride=1, padding=3)
        self.conv1_1_2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=(1, 7), stride=1)
        self.conv1_1_3 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, groups=int(in_channel / 2), padding=1)
        self.conv1_2_1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=(5, 1), stride=1, padding=2)
        self.conv1_2_2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=(1, 5), stride=1)
        self.conv1_2_3 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, groups=int(in_channel / 2), padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        
        x1 = self.conv1_1_1(x)
        x1 = self.conv1_1_2(x1)
        x1 = self.conv1_1_3(x1)
        output1 = x1
        x2 = self.conv1_2_1(x)
        x2 = self.conv1_2_2(x2)
        x2 = self.conv1_2_3(x2)
        output2 = x2
        output = output1 + output2
        output = self.relu(output)
        x = self.conv1_1_1(output)
        x = self.conv1_1_2(x)
        x = self.conv1_1_3(x)
        output1 = x
        x = self.conv1_2_1(output)
        x = self.conv1_2_2(x)
        x = self.conv1_2_3(x)
        output2 = x
        output = output1 + output2
        output = self.relu(output)
        return output

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 num_classes=1000,
                 include_top=True,
                 ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 32
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=(7, 1), stride=(2, 1), padding=3)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(1, 7), stride=(1, 2))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layer(block, 96)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)
    def _make_layer(self, block, channels):
        layer = []
        layer.append(block(channels))
        return nn.Sequential(*layer)
    def forward(self, x):
        # conv1-1
        x = self.conv1_1(x)
        x = self.relu(x)
        # conv1-2
        x = self.conv1_2(x)
        x = self.relu(x)
        # MaxPooling
        x = self.maxpool(x)
        # 残差块1(Residual block1)
        x = self.layer1(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        # conv3
        x = self.conv3(x)   
        x = self.relu(x)
        # 残差块2(Residual block2)
        x = self.layer2(x)
        # conv4
        x = self.conv4(x)
        x = self.relu(x)
        # AveragePooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.fc(x)
        return x


def resnet_muti(num_classes=1000, include_top=True):
    
    return ResNet(BasicBlock, num_classes=num_classes, include_top=include_top)

if __name__ == '__main__':
    import numpy as np
    a = np.ones((3, 224, 224), dtype=np.float32)[None, ...]
    a = torch.from_numpy(a)
    model = resnet_muti(num_classes=2)
    output = model(a)
    print(output)