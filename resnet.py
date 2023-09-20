# coding: utf-8
# In[ ]:
import torch
import torch.nn as nn #神经网络模块（nn）
import torch.nn.functional as F #激活函数（F）
import torch.optim as optim #优化器（optim）
import torch.optim.lr_scheduler as lr_scheduler #学习率调度器（lr_scheduler）
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, sampler, random_split #数据加载工具（DataLoader、sampler、random_split）
#这些导入与 PyTorch 及其组件有关

import torchvision.transforms as transforms #图像转换（transforms
import torchvision.datasets as datasets #预构建数据集（datasets）
import torchvision.models as models #模型（models）
#这些导入与 torchvision 库有关，该库提供了常用于计算机视觉任务的

from sklearn import decomposition
from sklearn import manifold #降维方法（decomposition、manifold）
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay #评估指标，如混淆矩阵（confusion_matrix、ConfusionMatrixDisplay）
#这些导入与 scikit-learn 相关，它是一个流行的机器学习库。
import matplotlib.pyplot as plt #Matplotlib 进行绘图（plt）
import numpy as np #使用 NumPy 处理数值数组（np）


import copy #复制对象（copy）
from collections import namedtuple #处理命名元组（namedtuple）
import os #操作系统功能（os）
import random #生成随机数（random）
import shutil #执行文件和目录操作（shutil）
import time #计时（time）
#通用的 Python 模块


# In[ ]:


SEED = 1234
#设置了一个种子（SEED）用于随机数生成器的初始化。通过设置相同的种子，可以确保在每次运行代码时得到相同的随机结果，这对于实验的可重现性非常重要。

random.seed(SEED) #设置 Python 内置的随机数生成器的种子
np.random.seed(SEED) #设置 NumPy 的随机数生成器的种子
torch.manual_seed(SEED) #设置 PyTorch 的随机数生成器的种子。
torch.cuda.manual_seed(SEED) #设置 PyTorch 的 CUDA 随机数生成器的种子。
torch.backends.cudnn.deterministic = True
#设置 PyTorch 的 cuDNN 库的行为为确定性模式，确保在相同输入情况下，每次运行的结果都是一致的。





#get_ipython().system(' pip install -q kaggle')

#from google.colab import files

#files.upload()

get_ipython().system(' mkdir ~/.kaggle #创建 ~/.kaggle 目录')

get_ipython().system(' cp kaggle.json ~/.kaggle/ #将 kaggle.json 文件复制到该目录中')

get_ipython().system(' chmod 600 ~/.kaggle/kaggle.json #并设置正确的文件权限')

get_ipython().system(' kaggle datasets list #使用 kaggle datasets list 命令来列出可用的数据集')


# In[ ]:

get_ipython().system('kaggle datasets download -d veeralakrishna/200-bird-species-with-11788-images')

get_ipython().system(' unzip  /content/200-bird-species-with-11788-images.zip #解压文件')


# In[ ]:


import tarfile

# 解压文件到/content目录下
tar = tarfile.open('/content/CUB_200_2011.tgz')
tar.extractall('/content')
tar.close()
#使用CUB200数据集的2011版本。这是一个包含200种不同鸟类的数据集。
#每个物种大约有60张图像，每张图像的大小约为500x500像素。


# In[ ]:


#该函数用于获取数据加载器和数据集的划分。它接受两个参数：data_dir（数据集所在的目录）和batch_size（批量大小）
def get_data_loaders(data_dir, batch_size):
  #transform = transforms.Compose([transforms.Resize(255),
  #                              transforms.CenterCrop(224),
  #                              transforms.ToTensor()])
  transform = transforms.Compose([transforms.Resize(256),
                   transforms.CenterCrop(224),
                   transforms.ToTensor()])
  all_data = datasets.ImageFolder(data_dir, transform=transform)
  #使用ImageFolder类加载数据集。该类假设数据集的目录结构按类别划分，并根据目录结构自动标注类别
  train_data_len = int(len(all_data)*0.75)
  valid_data_len = int((len(all_data) - train_data_len)/2)
  test_data_len = int(len(all_data) - train_data_len - valid_data_len)
  train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  #使用DataLoader类创建训练集、验证集和测试集的数据加载器。数据加载器用于批量加载数据，并提供数据的迭代器。
  #shuffle=True时，数据加载器会在每个时期重新洗牌数据，以增加训练的随机性。这对于避免模型过度拟合训练数据、增加模型的泛化能力非常重要。
  return ((train_loader, val_loader, test_loader),train_data, val_data, test_data, all_data.classes)
  #返回一个包含训练集、验证集和测试集的数据加载器的元组，以及训练集、验证集、测试集的数据集对象和类别信息。


# In[ ]:


(train_loader, val_loader, test_loader),train_data, val_data, test_data, classes = get_data_loaders("CUB_200_2011/images/", 64)
#调用get_data_loaders函数，传入数据集所在的目录路径"CUB_200_2011/images/"和批量大小64作为参数
#使用解包赋值的方式，将元组中的训练集、验证集和测试集的数据加载器分别赋值给train_loader、val_loader和test_loader变量。
#将数据集对象分别赋值给train_data、val_data和test_data变量。
#将类别信息赋值给classes变量。
#方便地访问和使用训练集、验证集和测试集的数据加载器、数据集对象以及类别信息。


# In[ ]:


#将图像进行归一化处理
def normalize_image(image):
    image_min = image.min()
    #图像张量中的最小值
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    #通过调用 clamp_ 方法，将图像张量中的值约束在 image_min 和 image_max 之间
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    #调用 add_ 方法将 image_min 从图像张量中减去，使得最小值变为 0
    #调用 div_ 方法将图像张量除以 image_max - image_min + 1e-5，进行归一化操作
    return image
    #最后，返回归一化后的图像张量
    #image为图像张量
#这些操作是就地修改图像张量的，而不是创建新的张量。
#这可以帮助节省内存，并且在训练神经网络时通常是一种常见的做法。


# In[ ]:


#这是一个名为 plot_images 的函数，用于绘制图像和对应的标签
#images, labels分别为图像列表和标签列表
def plot_images(images, labels, normalize = True):

    n_images = len(images)
    #n_images 变量记录了图像列表中的图像数量
    #images为图形列表

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    #计算绘图时所需的行数和列数，以便将图像按照方形布局排列。
    #图像数量的平方根取整，作为行数和列数。

    fig = plt.figure(figsize = (15, 15))
    #创建一个大小为 (15, 15) 的图形对象 fig

    #在循环中逐个处理图像并进行绘制
    for i in range(rows*cols):
    #循环迭代 rows*cols 次，即逐个遍历所有的子图位置。

        ax = fig.add_subplot(rows, cols, i+1)
        #创建一个子图，ax 是子图对象，并指定它在整个图形中的位置。i+1 表示子图的索引，从 1 开始。

        image = images[i]
        #从图像列表 images 中获取当前子图对应的图像。

        if normalize:
            image = normalize_image(image)
        #如果 normalize 为 True，则调用 normalize_image 函数对图像进行归一化处理

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        #ax 是一个子图对象，通过 fig.add_subplot() 创建。image 是当前子图对应的图像
        #image.permute(1, 2, 0) 用于重新排列图像张量的维度，以适应 imshow() 方法的要求。
        #通常，图像张量的维度顺序是 (通道数, 高度, 宽度)，而 imshow() 方法期望的维度顺序是 (高度, 宽度, 通道数)。
        #通过 permute() 方法，将通道维度调整为最后一个维度。
        #cpu().numpy() 将图像张量转换为 NumPy 数组。这是因为 imshow() 方法要求输入为 NumPy 数组格式。
        #最后，imshow() 方法会在子图中显示图像

        label = labels[i]
        #从标签列表 labels 中获取当前图像的标签
        ax.set_title(label)
        #设置子图的标题为当前图像的标签
        ax.axis('off')
        #关闭子图的坐标轴显示
#该函数引用了 normalize_image 函数，因此在调用 plot_images 函数之前，需要确保已经定义了 normalize_image 函数。
#该函数还使用了 numpy 和 matplotlib.pyplot 库。你需要确保已经导入了这些库。





# In[ ]:


#从训练数据中选择前 N_IMAGES 个样本，并将它们的图像和标签传递给 plot_images 函数进行绘制
N_IMAGES = 25
#要选择的图像数量

images, labels = zip(*[(image, label) for image, label in
                           [train_data[i] for i in range(N_IMAGES)]])
#列表推导式 [train_data[i] for i in range(N_IMAGES)] 用于从 train_data 中选择前 N_IMAGES 个样本
#(image, label) for image, label in ... 用于将每个样本的图像和标签解包为 (image, label) 对。
#zip(*...) 用于对解包后的图像和标签进行转置，以便得到 images 列表和 labels 列表，其中每个列表包含了对应位置的图像和标签。
plot_images(images, labels)
#plot_images(images, labels) 调用 plot_images 函数，将选择的图像和标签传递给它进行绘制。






# In[ ]:

class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h


# In[ ]:


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1,
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


# In[ ]:

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
#ResNetConfig是一个命名元组，通过namedtuple函数创建。
#它有三个字段：'block'、'n_blocks'和'channels'。这些字段分别表示块类、每个层中的块数量和每个层中的通道数。
#通过使用这个命名元组，我们可以更方便地定义和传递ResNet模型的配置信息


resnet18_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [2,2,2,2],
                               channels = [64, 128, 256, 512])

resnet34_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [3,4,6,3],
                               channels = [64, 128, 256, 512])


# In[ ]:


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1,
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


# In[ ]:


resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

resnet101_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 23, 3],
                                channels = [64, 128, 256, 512])

resnet152_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 8, 36, 3],
                                channels = [64, 128, 256, 512])


# In[ ]:


class CIFARResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        block, layers, channels = config
        self.in_channels = channels[0]

        assert len(layers) == len(channels) == 3
        assert all([i == j*2 for i, j in zip(channels[1:], channels[:-1])])

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self.get_resnet_layer(block, layers[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, layers[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, layers[2], channels[2], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):

        layers = []

        if self.in_channels != channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(channels, channels))

        self.in_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h


# In[ ]:


class Identity(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class CIFARBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            identity_fn = lambda x : F.pad(x[:, :, ::2, ::2],
                                           [0, 0, 0, 0, in_channels // 2, in_channels // 2])
            downsample = Identity(identity_fn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


# In[ ]:


cifar_resnet20_config = ResNetConfig(block = CIFARBasicBlock,
                                     n_blocks = [3, 3, 3],
                                     channels = [16, 32, 64])

cifar_resnet32_config = ResNetConfig(block = CIFARBasicBlock,
                                     n_blocks = [5, 5, 5],
                                     channels = [16, 32, 64])

cifar_resnet44_config = ResNetConfig(block = CIFARBasicBlock,
                                     n_blocks = [7, 7, 7],
                                     channels = [16, 32, 64])

cifar_resnet56_config = ResNetConfig(block = CIFARBasicBlock,
                                     n_blocks = [9, 9, 9],
                                     channels = [16, 32, 64])

cifar_resnet110_config = ResNetConfig(block = CIFARBasicBlock,
                                      n_blocks = [18, 18, 18],
                                      channels = [16, 32, 64])

cifar_resnet1202_config = ResNetConfig(block = CIFARBasicBlock,
                                       n_blocks = [20, 20, 20],
                                       channels = [16, 32, 64])


# In[ ]:


pretrained_model = models.resnet50(pretrained = True)
#使用torchvision库中的resnet50函数加载预训练的ResNet-50模型。
#设置pretrained=True将自动下载并加载预训练的权重参数。
#现在，pretrained_model将包含加载的预训练ResNet-50模型。


# In[ ]:


IN_FEATURES = pretrained_model.fc.in_features
#IN_FEATURES获取了预训练模型的最后一个线性层(fc)的输入特征维度，即2048。


OUTPUT_DIM = 200
#然后，OUTPUT_DIM被设置为200，以适应您的数据集中的类别数。
fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
#最后，使用nn.Linear创建了一个新的线性层fc，其输入特征维度为IN_FEATURES，输出特征维度为OUTPUT_DIM。
#这个新的线性层将用作预训练模型的最后一层，以适应您的数据集。


# In[ ]:


pretrained_model.fc = fc
#`pretrained_model`的最后一个线性层将被替换为我们新创建的线性层`fc`。
#这个新的线性层是随机初始化的，并且适应于我们的数据集的类别数。
#无论我们的数据集类别数是否与ImageNet相同，我们都会替换线性层，以确保模型输出与我们的数据集匹配。


# In[ ]:


model = ResNet(resnet50_config, OUTPUT_DIM)


# In[ ]:


model.load_state_dict(pretrained_model.state_dict())
#`state_dict`是一个字典对象，它保存了一个模型的参数（权重和偏置项）。
#`pretrained_model.state_dict()`返回预训练模型的参数字典，其中键是参数的名称，值是对应的参数张量。

#`model.load_state_dict()`方法将预训练模型的参数字典加载到自定义模型中，实现了参数的复制。
#加载参数时，它会确保自定义模型和预训练模型具有相同的参数键（keys），并将对应的参数值复制到自定义模型中。

#通过执行`model.load_state_dict(pretrained_model.state_dict())`，
#您的自定义模型`model`将获得与预训练模型相同的参数，从而初始化自定义模型的参数并使其与预训练模型相匹配。


# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[ ]:


START_LR = 1e-7
#定义了一个初始学习率 START_LR，设置为 1e-7，即 0.0000001。

optimizer = optim.Adam(model.parameters(), lr=START_LR)
#使用 Adam 优化器初始化了 optimizer，它将用于更新模型的参数。
#传递了 model.parameters()，以获取模型的所有可学习参数，并将初始学习率设置为 START_LR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#如果有可用的 CUDA 设备（GPU），则 device 被设置为 'cuda'，否则设备被设置为 'cpu'。
criterion = nn.CrossEntropyLoss()
#实例化了交叉熵损失函数 criterion。交叉熵损失通常用于多类别分类任务。
model = model.to(device)
criterion = criterion.to(device)
#将模型和损失函数都移动到指定的设备上。
#通过调用 .to(device) 方法，模型和损失函数的张量参数将被复制到指定设备的内存中，以便在该设备上进行计算。


# In[ ]:


#这段代码定义了一个LRFinder类，它实现了学习率范围测试的功能。
class LRFinder:
    def __init__(self, model, optimizer, criterion, device):

        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(model.state_dict(), 'init_params.pt')
    #在初始化方法中，首先保存了模型的初始参数状态(model.state_dict())到文件 'init_params.pt' 中。
    #然后，将传入的模型、优化器、损失函数和设备分配给类的成员变量。

    def range_test(self, iterator, end_lr = 10, num_iter = 100,
                   smooth_f = 0.05, diverge_th = 5):

        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)

        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            #update lr
            lr_scheduler.step()

            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        #reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))

        return lrs, losses
    #range_test 方法是学习率范围测试的核心部分。
    #它接受一个迭代器iterator作为输入数据源，并在指定的迭代次数num_iter内进行训练。
    #在每个迭代中，它调用_train_batch 方法来训练一个小批量样本，并更新学习率。
    #然后，它记录当前学习率和损失值，并根据平滑系数smooth_f平滑损失值。
    #如果当前损失值超过最佳损失值的一定倍数diverge_th，则停止训练。
    #最后，它加载初始参数状态，以便后续使用，并返回记录的学习率和损失值。

    def _train_batch(self, iterator):

        self.model.train()

        self.optimizer.zero_grad()

        x, y = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred, _ = self.model(x)

        loss = self.criterion(y_pred, y)

        loss.backward()

        self.optimizer.step()

        return loss.item()
    #_train_batch 方法用于训练一个小批量样本。
    #它将模型设为训练模式，将梯度清零，将输入数据和标签移动到指定设备上，然后计算预测值并计算损失。
    #接着，它进行反向传播和参数更新，并返回损失值

#ExponentialLR 类是一个学习率调度器，它根据指定的迭代次数和最终学习率，在每次调度时计算当前学习率。
#它继承自 _LRScheduler 类，并实现了 get_lr 方法来计算当前学习率。
class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)


# In[ ]:


END_LR = 10
NUM_ITER = 100
#定义了`END_LR`和`NUM_ITER`作为测试的结束学习率和迭代次数

lr_finder = LRFinder(model, optimizer, criterion, device)
#创建了一个`LRFinder`对象`lr_finder`，并将模型、优化器、损失函数和设备传递给它的构造函数
lrs, losses = lr_finder.range_test(train_loader, END_LR, NUM_ITER)
#调用`lr_finder`对象的`range_test`方法来执行学习率范围测试。
#传递了`train_loader`作为迭代器，`END_LR`作为结束学习率，`NUM_ITER`作为迭代次数
#`range_test`方法会在指定的迭代次数内进行训练，同时逐步增加学习率。
#它会记录每个迭代步骤的学习率和损失值，并在损失值发散时提前停止训练。
#最后，`lrs`和`losses`变量将包含学习率和损失值的列表，可以用于可视化学习率范围测试的结果。


# In[ ]:


#lrs表示学习率的列表，losses表示损失值的列表，skip_start和skip_end是可选参数，用于指定要跳过的起始和结束数据点的数量。
def plot_lr_finder(lrs, losses, skip_start = 5, skip_end = 5):

    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    #根据skip_start和skip_end的值来选择要绘制的数据点。
    #如果skip_end为0，表示不跳过任何结尾的数据点，那么从skip_start索引开始的所有数据点都将被保留。
    #否则，从skip_start索引开始，直到倒数第skip_end个索引结束的数据点将会被保留。

    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(1,1,1)
    #子图对象ax 创建了一个1x1的网格，并将子图放置在第一个（唯一一个）位置上。
    ax.plot(lrs, losses)
    #在子图ax上绘制了学习率和损失的折线图。
    ax.set_xscale('log')
    #设置横轴为对数刻度
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    #ax.grid()函数用于控制子图的网格线显示。
    #参数True表示要显示网格线，'both'表示网格线同时显示在x轴和y轴上，而'x'表示只在x轴上显示网格线。
    plt.show()


# In[ ]:


plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)
#使用给定的参数调用plot_lr_finder函数，将会绘制一个学习率范围测试的结果图表。该图表将跳过前30个数据点和后30个数据点
#这样做可以更好地聚焦于学习率范围测试的中间部分，以便观察到更具意义的变化和趋势。


# In[ ]:


FOUND_LR = 1e-3
#将学习率的值设置为1e-3，即0.001。这个学习率是根据之前的分析得出的最佳学习率。

params = [
          {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
          {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
          {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
          {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
          {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
          {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
          {'params': model.fc.parameters()}
          #fc层（全连接层）没有指定学习率。这意味着它将使用默认的学习率，即最大学习率 FOUND_LR
         ]


optimizer = optim.Adam(params, lr = FOUND_LR)
#创建了一个Adam优化器对象 optimizer，并将之前定义的参数列表 params 和最大学习率 FOUND_LR 作为参数传递给优化器。这将用于更新模型的参数，以最小化损失函数。


# In[ ]:


#使用PyTorch中的lr_scheduler.OneCycleLR来设置一个单周期学习率调度器。
EPOCHS = 10
STEPS_PER_EPOCH = len(train_loader)
#训练集的数据加载器中的批次数
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
#定义了总共的训练步数TOTAL_STEPS，通过将训练周期数EPOCHS乘以每个周期中的步数STEPS_PER_EPOCH计算得到

MAX_LRS = [p['lr'] for p in optimizer.param_groups]
#获取了优化器中每个参数组的最大学习率MAX_LRS。
#optimizer.param_groups返回了优化器中的参数组列表，每个参数组包含了一组参数及其对应的学习率等信息。
#通过遍历参数组列表，将每个参数组的学习率lr添加到MAX_LRS列表中

scheduler = lr_scheduler.OneCycleLR(optimizer,
                   max_lr = MAX_LRS,
                   total_steps = TOTAL_STEPS)
#使用lr_scheduler.OneCycleLR创建了一个单周期学习率调度器scheduler。传入了优化器optimizer、最大学习率列表max_lr，以及总步数total_steps作为参数。
#这样设置的单周期学习率调度器将根据总步数来动态调整学习率。学习率会从最小值逐渐增加到对应参数组的最大学习率，然后再逐渐减小到较小的最终学习率。这个调度器会在每次参数更新步骤之后更新学习率。
#根据代码中的MAX_LRS列表，这个单周期学习率调度器将对每个参数组使用不同的最大学习率，实现了区分微调（discriminative fine-tuning）的效果。
#如果只传递单个学习率而不是学习率列表，调度器将应用相同的学习率于所有参数组，没有区分微调的效果。


# In[ ]:


#计算Top-k准确率
def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        #correct_1 = correct[:1].view(-1).float().sum(0, keepdim = True)
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k


# In[ ]:


#输入包括模型（model）、迭代器（iterator）、优化器（optimizer）、损失函数（criterion）、调度器（scheduler）和设备（device）
def train(model, iterator, optimizer, criterion, scheduler, device):

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    #初始化了每个训练周期的训练损失（epoch_loss）、Top-1准确率（epoch_acc_1）和Top-5准确率（epoch_acc_5），并将它们都设为0
    model.train()
    #调用 model.train() 将模型设置为训练模式
    for (x, y) in iterator:
    #训练循环遍历由迭代器提供的数据。每次迭代处理一个批次的输入（x）和标签（y）
        x = x.to(device)
        y = y.to(device)
        #使用 x.to(device) 和 y.to(device) 将输入和标签移动到指定的设备
        optimizer.zero_grad()
        #使用 optimizer.zero_grad() 清除优化器的梯度
        y_pred, _ = model(x)
        #使用模型对输入进行预测（y_pred）
        loss = criterion(y_pred, y)
        #使用指定的损失函数 criterion(y_pred, y) 计算损失
        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
        #使用 calculate_topk_accuracy 函数计算Top-1和Top-5准确率
        loss.backward()
        #通过调用 loss.backward() 计算梯度
        optimizer.step()
        #使用 optimizer.step() 更新模型参数
        scheduler.step()
        #使用 scheduler.step() 更新学习率
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        #累加并平均每个训练周期的训练损失、Top-1准确率和Top-5准确率
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_5
    #最后，函数返回训练周期的平均训练损失、Top-1准确率和Top-5准确率作为输出


# In[ ]:


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    #初始化了每个评估周期的评估损失（epoch_loss）、Top-1准确率（epoch_acc_1）和Top-5准确率（epoch_acc_5），并将它们都设为0。
    model.eval()

    with torch.no_grad():
    #使用 torch.no_grad() 上下文管理器，禁止梯度计算。
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_5
    #最后，函数返回评估周期的平均评估损失、Top-1准确率和Top-5准确率作为输出。


# In[ ]:


#输入是开始时间（start_time）和结束时间（end_time）
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
#通过计算 end_time - start_time 得到经过的时间。
#将经过的时间转换为分钟数（elapsed_mins）和秒数（elapsed_secs）。
#最后，函数返回经过的分钟数和秒数作为输出。


# In[ ]:


#代码是一个训练循环，用于训练和评估模型，并在验证损失达到最佳时保存模型。
#打印每个训练周期的信息
best_valid_loss = float('inf')
#将 best_valid_loss 初始化为正无穷，用于跟踪最佳的验证损失
for epoch in range(EPOCHS):
#对于每个训练周期，执行以下操作：
    start_time = time.time()
    #记录当前训练周期的起始时间。
    train_loss, train_acc_1, train_acc_5 = train(model, train_loader, optimizer, criterion, scheduler, device)
    #调用 train 函数进行模型训练，并将返回的训练损失、Top-1训练准确率和Top-5训练准确率分别赋值给变量 train_loss、train_acc_1 和 train_acc_5。
    valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, val_loader, criterion, device)
    #调用 evaluate 函数对模型进行评估，并将返回的验证损失、Top-1验证准确率和Top-5验证准确率分别赋值给变量 valid_loss、valid_acc_1 和 valid_acc_5。

    #如果当前验证损失小于最佳验证损失 best_valid_loss，则执行以下操作：
    #更新最佳验证损失为当前验证损失。保存模型的状态字典到文件 "tut5-model.pt"。
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut5-model.pt')

    end_time = time.time()
    #记录当前训练周期的结束时间
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
          f'Train Acc @5: {train_acc_5*100:6.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
          f'Valid Acc @5: {valid_acc_5*100:6.2f}%')
    #打印训练周期的信息，包括周期编号、耗时、训练损失和准确率，以及验证损失和准确率
    #格式化字符串中使用 :02 表示打印两位数的周期编号，:.3f 表示打印三位小数的损失值，:6.2f 表示打印六位宽度、两位小数的准确率值。


# In[ ]:


#加载保存的模型，并在测试集上进行评估并打印测试结果。
model.load_state_dict(torch.load('tut5-model.pt'))
#加载之前保存的模型状态字典，将其恢复为训练过程中保存的最佳模型。
test_loss, test_acc_1, test_acc_5 = evaluate(model, test_loader, criterion, device)
#调用 evaluate(model, test_loader, criterion, device) 对加载的模型在测试集上进行评估。
#返回的测试损失、Top-1测试准确率和Top-5测试准确率分别赋值给变量 test_loss、test_acc_1 和 test_acc_5。
print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
      f'Test Acc @5: {test_acc_5*100:6.2f}%')


# In[ ]:


#定义了 get_predictions 函数，该函数接受一个模型和一个数据迭代器作为输入，并返回图像、标签和概率的张量。
def get_predictions(model, iterator):

    model.eval()
    #调用 model.eval() 将模型设置为评估模式，以确保在推理过程中不进行梯度计算

    images = []
    labels = []
    probs = []

    with torch.no_grad():
    #使用 torch.no_grad() 上下文管理器来禁用梯度计算，以减少内存消耗和加速推理过程

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)
            #遍历数据迭代器中的每个批次，将输入数据移动到适当的设备上，并使用模型进行前向传播，得到预测结果 y_pred

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)
            #将预测结果应用 softmax 函数以获得概率分布 y_prob，并使用 argmax 函数找到最高概率对应的类别索引 top_pred。

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
            #最后，我们将每个批次中的图像、标签和概率结果分别添加到对应的列表中

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    #遍历完成后，我们使用 torch.cat 函数将列表中的张量沿指定维度进行拼接，得到完整的图像、标签和概率张量

    return images, labels, probs
    #最后，我们返回拼接后的图像、标签和概率张量作为函数的输出


# In[ ]:


images, labels, probs = get_predictions(model, test_loader)
#给定的模型和 test_loader，你可以通过以下代码获取图像、标签和概率的张量


# In[ ]:


pred_labels = torch.argmax(probs, 1)
#torch.argmax 函数用于在指定的维度上找到张量中最大值的索引。
#对于给定的概率张量 probs，你可以使用以下代码获取预测的标签


# In[ ]:


#len(labels)#真实标签
#len(pred_labels)#预测标签
#len(images)


# In[ ]:


#绘制混淆矩阵
import seaborn as sns
sns.set(font="SimHei", font_scale=1.5)
from sklearn.metrics import confusion_matrix as CM
y_pred_SVC=pred_labels

SVM=CM(labels,y_pred_SVC)

plt.figure(figsize=(10,8),dpi=200)
#fig,ax=plt.subplots(3,3,figsize=(10,10),dpi=200)
sns.set(font="SimHei",font_scale=1.0)

font={'family':'Times New Roman',
    'weight':'normal',
      'size':25
}

ax4=plt.subplot()
#ns.heatmap(SVM,ax=ax[1][0],annot=True,cmap=plt.cm.GnBu)
sns.heatmap(SVM,ax=ax4,annot=True,cmap=plt.cm.GnBu)
ax4.set_title("SVM Confusion matrix",fontdict=font)
ax4.set_xticklabels(['',''],fontsize=10,rotation=90)
ax4.set_yticklabels(['',''],fontsize=10,rotation=360);

#plt.savefig('/scripts/SVM-CM.jpg')

