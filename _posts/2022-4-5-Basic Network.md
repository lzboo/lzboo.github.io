---
layout: article
title: Document - Writing Posts
mathjax: true
---

# 一、实验内容
        在Fashio-MNist数据集上实现:softmax,MLP,LeNet,AlexNet,GooLeNet,ResNet模型训练和测试，对模型的性能进行比对分析。


# 二、实验环境
* **环境**：Colab
* **框架**：Pytorch

# 三、Fashion-MNist数据集
## 3.1 介绍
* 类别数：10
* 样本：图片，feature.shape = [1, 28, 28] (channel, weight, high)
* 训练集和测试集中每个类别图像数分别位6000和1000，故训练集和测试集样本数为60000和10000
## 3.2 数据集获取
```python
Fmnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())

Fmnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
```

# 四、相关函数定义
## 4.1 读取小批量数据
```python
def load_data_fashion_mnist1(batch_size):
  if sys.platform.startswith('win'):
      num_workers = 0  # 0表示不用额外的进程来加速读取数据
  else:
      num_workers = 4
  train_iter = torch.utils.data.DataLoader(Fmnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  test_iter = torch.utils.data.DataLoader(Fmnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  return train_iter, test_iter
  
# mnist_train是torch.utils.data.Dataset的子类，
# 所以我们可以将其传入torch.utils.data.DataLoader来创建一个读取小批量数据样本的DataLoader实例。
```
```python
def load_data_fashion_mnist2(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter
```

## 4.2 评价模型net的准确率
```python
def evaluate_accuracy1(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
```
```python
def evaluate_accuracy2(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: 
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n
```
## 4.3 模型训练
```python
def train_model1(net, train_iter, test_iter, loss, num_epochs, batch_size, patams=None, lr=None, optimizer=None):

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                param.grad.data.zero_()

            # 优化
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]

        loss_train = train_l_sum / n   # 失误1： 变量重新定义，直接loss = ... 报错：'float' object is not callable
        train_acc = train_acc_sum / n
        test_acc = evaluate_accuracy(test_iter, net)

        print('epoch %d, loss: %.4f train acc: %.3f test acc: %.3f' %(epoch+1, loss_train, train_acc, test_acc))
```

```python
def train_model2(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
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

        # 梯度清零+优化
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        loss_train = train_l_sum / batch_count   # 失误1： 变量重新定义，直接loss = ... 报错：'float' object is not callable
        train_acc = train_acc_sum / n
        test_acc = evaluate_accuracy(test_iter, net)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'% (epoch + 1, loss_train, train_acc, test_acc, time.time() -start))

```
## 4.4 送入全连接层前对x形状转换
```python
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)  # 将向量展平方便送入全连接层

# 数据返回的每个batch样本x的形状为(batch_size, 1, 28, 28)
# 所以我们要先用view()将x的形状转换成(batch_size, 784)才送入全连接层。
```
## 4.5 全局池化层
```python
import torch.nn.functional as F
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
```

# 五、模型原理及训练
## 5.1 Softmax
### 5.1.1 相关
* softmax是多分类模型，且引入softmax运算使输出更适合离散值预测和训练。
* softmax回归同线性回归一样，也是一个单层神经网络。
* softmax回归跟线性回归一样将输入特征与权重做线性叠加。与线性回归的一个主要不同在于，softmax回归的输出值个数等于标签里的类别数。
* 损失函数：交叉熵损失（cross entropy）
    * 最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。
* Cons:
    * softmax回归适用于分类问题，使用softmax运算输出类别的概率分布。
    * softmax回归是一个单层神经网络，输出个数等于分类问题中的类别个数。
    * 交叉熵适合衡量两个概率分布的差异。
* softmax回归的输出值个数等于标签里的类别数。因为一共有4种特征和3种输出动物类别，所以权重包含12个标量w、偏差包含3个标量b，且对每个输入计算o1,o2,o3o:
![](softmax_1.png)
![](softmax_2.png)
![](softmax_3.png)

### 5.1.2 实现
**1. 读取数据**
```python
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist1(batch_size)
```
**2. 定义和初始化模型**
```python
num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)

# 定义模型
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_outputs)
)

# 初始化参数w,b
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0) 
```
**3. 损失函数：交叉熵**
```python
loss = nn.CrossEntropyLoss()
```
**4. 优化算法：小批量随机梯度下降**
```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
```
**5. 模型训练**
```python
num_epochs = 5
train_model(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

## 5.2 MLP
### 5.2.1 相关
* Cons:
    * 多层感知机在输出层与输入层之间加入了一个或多个全连接隐藏层，并通过激活函数对隐藏层输出进行变换。
    * 常用的激活函数包括ReLU函数、sigmoid函数和tanh函数。

### 5.2.2 实现
**1. 定义模型**
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )

# 使用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
```
**2. 读取数据并训练**
```python
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist1(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
train_model1(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```
### 5.2.3 结果
```
epoch 1, loss 0.0031, train acc 0.706, test acc 0.744
epoch 2, loss 0.0019, train acc 0.817, test acc 0.796
epoch 3, loss 0.0017, train acc 0.842, test acc 0.797
epoch 4, loss 0.0015, train acc 0.857, test acc 0.779
epoch 5, loss 0.0014, train acc 0.865, test acc 0.822
```
## 5.3 LeNet
### 5.3.1 LeNet模型
**1. 模型结构**
![](lenet_1.png)
![](lenet_2.png)
```python
LeNet(
  (conv): Sequential(
    (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): Sigmoid()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Sigmoid()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=256, out_features=120, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=120, out_features=84, bias=True)
    (3): Sigmoid()
    (4): Linear(in_features=84, out_features=10, bias=True)
  )
)
```
### 5.3.2 实现
**1. 模型定义**
```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```
**2. 数据获取与模型训练**
```python
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_model2(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```

### 5.3.3 结果
```
epoch 1, loss 1.9525, train acc 0.273, test acc 0.556, time 8.4 sec
epoch 2, loss 0.9631, train acc 0.633, test acc 0.688, time 8.3 sec
epoch 3, loss 0.7502, train acc 0.724, test acc 0.732, time 8.2 sec
epoch 4, loss 0.6645, train acc 0.748, test acc 0.752, time 8.2 sec
epoch 5, loss 0.6076, train acc 0.766, test acc 0.759, time 8.4 sec
```

> [1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.


## 5.4 AlexNet
### 5.4.1 原理
**1. 模型结构**
![](Alexnet_1.png)
![](Alexnet_2.jpg)
```python
AlexNet(
  (conv): Sequential(
    (0): Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
    (8): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU()
    (10): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU()
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=6400, out_features=4096, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
)
```
### 5.4.2 实现
**1. 模型定义**
```python
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```
**2. 数据获取和模型训练**
```python
batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist2(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_model2(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```
### 5.4.3 结果
```
epoch 1, loss 0.2948, train acc 0.891, test acc 0.896, time 169.3 sec
epoch 2, loss 0.2596, train acc 0.902, test acc 0.900, time 169.1 sec
epoch 3, loss 0.2391, train acc 0.910, test acc 0.903, time 168.9 sec
```

> [1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
## 5.5 VGG
### 5.5.1 原理
**1. 网络结构**
![](VGG_2.png)
```python
Sequential(
  (vgg_block_1): Sequential(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_2): Sequential(
    (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_3): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_4): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (vgg_block_5): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): FlattenLayer()
    (1): Linear(in_features=3136, out_features=512, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=512, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.5, inplace=False)
    (7): Linear(in_features=512, out_features=10, bias=True)
  )
)
```
### 5.5.2 实现
**1. VGG块**
```python
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)
```
**2. VGG网络**
```python
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                    nn.Linear(fc_features, fc_hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(fc_hidden_units, fc_hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                   nn.Linear(fc_hidden_units, 10)
                  ))
    return net
```
**3. 数据获取与模型训练**
```python
ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)

batch_size = 64
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_model2(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```
### 5.5.3 结果
```
epoch 1, loss 0.5624, train acc 0.790, test acc 0.878, time 103.2 sec
epoch 2, loss 0.3261, train acc 0.881, test acc 0.895, time 102.3 sec
epoch 3, loss 0.2785, train acc 0.898, test acc 0.905, time 103.2 sec
epoch 4, loss 0.2494, train acc 0.910, test acc 0.908, time 102.6 sec
epoch 5, loss 0.2239, train acc 0.918, test acc 0.915, time 102.7 sec
```

> [1] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
## 5.6 NiN
### 5.6.1 原理
**1. 网络结构**
![](NiN_1.png)
![](NiN_2.jpg)
### 5.6.2 实现
**1. NIN块**
```python
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk
```
**2. NIN网络**
```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(), 
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    FlattenLayer())
```
**3. 数据获取与模型训练**
```python
batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist2(batch_size, resize=224)

lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_model2(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```

### 5.6.3 结果
```
epoch 1, loss 1.5475, train acc 0.411, test acc 0.749, time 177.3 sec
epoch 2, loss 0.6163, train acc 0.769, test acc 0.795, time 176.6 sec
epoch 3, loss 0.5060, train acc 0.813, test acc 0.825, time 176.5 sec
epoch 4, loss 0.4509, train acc 0.834, test acc 0.842, time 176.6 sec
epoch 5, loss 0.4170, train acc 0.847, test acc 0.849, time 176.3 sec
```

> [1] Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.
## 5.7 GooleNet
### 5.7.1 原理
![](gooLenet_1.png)

### 5.7.2 实现
**1. Inception块**
```python
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出
```
**2. GooLeNet网络**
```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5, 
                    FlattenLayer(), nn.Linear(1024, 10))
```
**3. 数据获取与模型训练**
```python
batch_size = 128

train_iter, test_iter = load_data_fashion_mnist2(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_model2(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```

### 5.7.3 结果
```
epoch 1, loss 1.0778, train acc 0.580, test acc 0.793, time 162.0 sec
epoch 2, loss 0.4220, train acc 0.843, test acc 0.845, time 161.8 sec
epoch 3, loss 0.3441, train acc 0.870, test acc 0.871, time 161.9 sec
epoch 4, loss 0.3041, train acc 0.886, test acc 0.881, time 161.8 sec
epoch 5, loss 0.2792, train acc 0.894, test acc 0.891, time 161.4 sec
```

> [1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., & Anguelov, D. & Rabinovich, A.(2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
## 5.8 Resnet
### 5.8.1 原理
![](Resnet_1.png)

### 5.8.2 实现
**1. 残差块**
```python
class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
```
**2. 残差网络**
```python
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 对模块1特别处理
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# 为ResNet加入所有残差块
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

# 加入全局平均池化层后接上全连接层输出
net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10))) 

```
**3. 数据获取与模型训练**
```python
batch_size = 256
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist2(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_model2(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
```
### 5.8.3 结果
```
epoch 1, loss 0.4110, train acc 0.847, test acc 0.865, time 41.3 sec
epoch 2, loss 0.2586, train acc 0.904, test acc 0.907, time 42.0 sec
epoch 3, loss 0.2118, train acc 0.922, test acc 0.910, time 42.9 sec
epoch 4, loss 0.1851, train acc 0.931, test acc 0.912, time 43.0 sec
epoch 5, loss 0.1616, train acc 0.940, test acc 0.920, time 43.0 sec
```

>[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
---

# 六、问题探讨
## 6.1 网络深度与参数量对模型性能影响
## 6.2 批量大小（batch_size）对模型影响

> 非常感谢李沐大神，致敬！
