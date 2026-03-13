import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_datasets(data_dir='./data', download=True):
    """
    获取MNIST数据集
    
    Args:
        data_dir: 数据存储目录
        download: 是否下载数据集
    
    Returns:
        train_dataset, test_dataset: 训练和测试数据集
    """
    # 数据预处理：转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 下载并加载训练数据集
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    # 下载并加载测试数据集
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )
    
    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size=64):
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        batch_size: 批次大小
    
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader