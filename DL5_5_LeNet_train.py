import time
import torch
from torch import optim
import DL3_5_download_FashionMNIST as dl

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n



def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        

"""Notes
1. if device is None and isinstance(net, torch.nn.Module):
device = list(net.parameters())[0].device 
这段代码检查了是否有使用指定设备,若无,会自动检测模型实例所在设备
isinstance(net, torch.nn.Module)检查net是否是Pytorch模型
list(net.parameters())[0].device获取模型第一个参数所在设备
→ 正确评估在(CPU/GPU)上进行
2. 迭代过程中的if isinstance(net, torch.nn.Module) → 
检查net是否是nn.Module类实例, 如果是就用pytroch标准方法处理
如果不是 → 检查函数是否有is_training函数,有则在评估时设置为False
3. net = net.to(device)将神经网络移到指定设备(CPU/GPU)
4. X = X.to(device) / y = y.to(device)
将输入数据X和标签y移动到与模型相同的设备上
5. l.cpu().item()将损失值从GPU转移到CPU,并转换为python标量
确保最终标量值在CPU上 ∵pytorch的张量.item()方法只能在CPU上用
"""













