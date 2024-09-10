"""卷积层的两大特点
1. 保留输入形状 → 识别空间相关性
(1)二维结构保留：保留了二维数据结构，处理的时候不需要展平为一维向量 → 展平会丢失空间信息
(2)局部连接：卷积核只和输入的局部区域连接 → 模拟出生物视觉系统的感受野，可以有效捕捉边缘、纹理等局部特征
(3)空间相关性：卷积操作能同时考虑像素在水平和垂直方向的关系 vital for 识别复杂视觉模式

2. 参数共享，避免参数量过大
滑动窗口机制以为着同样的特征检测器应用于输入的所有位置 带来多个优势
(1)图像特征即使产生平移，仍能被相同卷积捕捉到特征 → 平移不变性
(2)参数减少，卷积核中参数重复利用
(3)参数共享→计算效率提高
(4)参数减少和局部连接性有助于减少过拟合 → 提高模型泛化能力
(5)改变卷积核大小/不同池化操作 → 多尺度特征提取"""

import time 
import torch
from torch import nn, optim
import DL3_5_download_FashionMNIST as dl
from DL5_5_LeNet_model import LeNet
from DL5_5_LeNet_train import evaluate_accuracy, train_ch5
import sys 
sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')




def main():
    batch_size = 256
    lr, num_epochs = 0.001, 100

    train_iter, test_iter = dl.load_data_fashion_mnist(batch_size=batch_size)

    net = LeNet()
    print(net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

if __name__ == '__main__':
    main()
