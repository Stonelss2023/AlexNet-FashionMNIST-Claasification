import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    

"""Notes
1. 从纯数学的角度来说如果要保持总元素数量不变,第二层卷积增加8个通道
确实是更精确的做法, 但是保持总元素量相等并非网络的内在要求, 网络设计
理应更关注：
(1)逐渐增加特征的抽象层次
(2)在减少空间分辨率的同时增加通道数
(3)平衡网络表达能力和计算复杂度

1998年诞生的LeNet设计有其历史局限性,更多是基于直觉和实验的设计
而非严格的数学推导

2. nn.Sequenstial()的作用 → 一个有序容器,用于按顺序堆叠神经网络层
允许我们将多个层组合在一起,形成一个更大的网络模块
当数据通过Sequential时,会按照定义的顺序一次通过每一层

完美避免在forward方法中写出每一层的操作,自动处理层与层之间的连接
传统方法需要在__init__中单独定义每一层,使用Sequentail可以在__init__中一次性定义整个网络结构

3. 代码结构————借助Sequential,网络被分为 卷积 和 全连接 两个部分,内部使用Sequential 来组织多个层

4. 当使用raw FashionMNIST数据集输入的时候, 经过LeNet的网络加工最后一层输出的特征图正好是4*4!!

5. 仍然需要定义一个forward()函数 → 定义数据在网络中的流动路径+允许层之间的自定义操作"""