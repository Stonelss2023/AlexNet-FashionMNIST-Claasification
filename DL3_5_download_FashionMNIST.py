import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
import torch.utils.data as Data
sys.path.append("..")

def load_data_fashion_mnist(batch_size, resize=None): #resize → adjust the size of the images
    mnist_train = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=True, download=False,
                                                    transform=transforms.ToTensor())
    
    mnist_test = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                   train=False, download=False,
                                                   transform=transforms.ToTensor())
      

    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size,
                                                 shuffle=True)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size,
                                                shuffle=False)
    return train_iter, test_iter


"""Notes
1. sys.path.append("..") → 将当前目录的父目录添加到Python模块的搜索路径中
允许导入位于父目录的模块 → 通常用于组织较大的项目，使不同目录下的模块可以相互引用
2. torchvision → pytorch专门用于cv任务的配套库
datasets 是torchvision中的一个模块 → 提供很多预定义的数据集(含FashionMNIST)
FashionMNIST已经预定义好了,包括图像标准化和标签编码。用torchvision.datasets
可以大大简化数据加载过程 → 专注于模型设计和训练
DataLoader 进一步简化batch处理,打乱和并行加载数据过程
3. transforms.ToTensor() 是一个转换函数
(1) 将PIL Image 或 Numpy ndarray 转换为 Pytroch tensor
(2) 将像素值范围懂[0,255] 缩放到 [0.0,1.0]
4. 导入参数代码分行写 → 提高可读性
5. 作为关键字参数 train= 和 download= 的位置是可以调换的"""





