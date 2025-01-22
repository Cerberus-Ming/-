import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约160M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化 先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='DataSet/',  # 将下载的数据集压缩包解压到当前目录的DataSet目录下
    train=True,
    download=True,  # 如果之前没手动下载数据集，这里要改为True
    transform=transform)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
    'DataSet/',
    train=False,
    download=True,  # 如果之前没手动下载数据集，这里要改为True
    transform=transform)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[66]
data.size()  # 验证某一张图片的维度 —— 3*32*32
print(classes[label])  # label是一个0-9的数字
# (data + 1) / 2是为了还原被归一化的数据 （这部分计算是可以推算出来的）
show((data + 1) / 2).resize((100, 100))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 最后是一个十分类，所以最后的一个全连接层的神经元个数为10

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # 展平  x.size()[0]是batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

from torch import optim

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降

t.set_num_threads(4)
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):  # i 第几个batch     data：一个batch中的数据

        # 输入数据
        inputs, labels = data  # images：batch大小为4     labels：batch大小为4

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()  # 一个batch返回4张图片
print('实际的label: ', ' '.join(
    '%08s' % classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100))
show(tv.utils.make_grid((images + 1) / 2)).resize((400, 100))

images.shape
# 计算图片在每个类别上的分数
outputs = net(images)  # images 4张图片的数据
# 得分最高的那个类
_, predicted = t.max(outputs.data, 1)  # 1是维度参数，返回值为两个，一个为最大值，另一个为最大值的索引

print('预测结果: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数

# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with t.no_grad():
    for data in testloader:      # data是个tuple
        images, labels = data    # image和label 都是tensor
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)    # labels tensor([3, 8, 8, 0])            labels.size: torch.Size([4])
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with t.no_grad():
    for data in testloader:      # data是个tuple
        images, labels = data    # image和label 都是tensor
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)    # labels tensor([3, 8, 8, 0])            labels.size: torch.Size([4])
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
