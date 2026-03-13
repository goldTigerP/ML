import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    多层感知机（MLP）模型用于MNIST手写数字识别
    """
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 将输入图像展平为一维向量
        x = x.view(x.size(0), -1)
        
        # 第一层 + ReLU激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 第二层 + ReLU激活 + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    """
    卷积神经网络（CNN）模型用于MNIST手写数字识别
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.conv1(x)))
        
        # 第二个卷积块
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平特征图
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x