# MNIST手写数字识别项目

这是一个使用PyTorch实现的MNIST手写数字识别项目，包含MLP（多层感知机）和CNN（卷积神经网络）两种模型。

## 项目结构

```
mnist_recognition/
├── main.py                 # 主程序入口
├── requirements.txt        # 项目依赖
├── README.md              # 项目说明
├── src/                   # 源代码目录
│   ├── models.py          # 神经网络模型定义
│   ├── data_utils.py      # 数据处理工具
│   ├── trainer.py         # 训练器类
│   └── utils.py           # 工具函数
├── models/                # 保存训练好的模型
├── data/                  # MNIST数据集存储目录
└── outputs/               # 输出文件（图表、预测结果等）
```

## 功能特点

- **两种模型架构**：
  - MLP（多层感知机）：简单的全连接网络
  - CNN（卷积神经网络）：更适合图像识别的卷积网络

- **完整的训练流程**：
  - 自动下载MNIST数据集
  - 数据预处理和增强
  - 模型训练和验证
  - 训练过程可视化
  - 模型保存和加载

- **丰富的分析工具**：
  - 训练历史可视化
  - 预测结果可视化
  - 错误分析和混淆矩阵
  - 模型结构摘要

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速训练（使用默认参数）

```bash
python main.py
```

这将使用默认参数训练一个MLP模型，训练10个epoch。

### 3. 自定义训练

```bash
# 训练MLP模型
python main.py --model mlp --epochs 20 --batch_size 128 --lr 0.001

# 训练CNN模型
python main.py --model cnn --epochs 15 --batch_size 64 --lr 0.0005
```

### 4. 测试已训练的模型

```bash
python main.py --mode test --model mlp
```

### 5. 演示预测结果

```bash
python main.py --mode demo --model mlp
```

## 命令行参数

- `--model`: 选择模型类型 (`mlp` 或 `cnn`)
- `--epochs`: 训练轮数 (默认: 10)
- `--batch_size`: 批次大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.001)
- `--no_cuda`: 禁用CUDA加速
- `--mode`: 运行模式 (`train`, `test`, 或 `demo`)

## 模型架构

### MLP模型
- 输入层: 784个节点 (28x28像素)
- 隐藏层1: 512个节点 + ReLU + Dropout
- 隐藏层2: 512个节点 + ReLU + Dropout
- 输出层: 10个节点 (0-9数字分类)

### CNN模型
- 卷积层1: 1→32通道, 3x3卷积核
- 最大池化层1: 2x2池化
- 卷积层2: 32→64通道, 3x3卷积核
- 最大池化层2: 2x2池化
- 全连接层1: 3136→128 + ReLU + Dropout
- 全连接层2: 128→10

## 预期性能

- **MLP模型**: 测试准确率约97-98%
- **CNN模型**: 测试准确率约98-99%

## 输出文件

训练完成后会生成以下文件：

- `models/best_model.pth`: 最佳模型权重
- `training_history.png`: 训练历史曲线
- `predictions_visualization.png`: 预测结果可视化
- `confusion_matrix.png`: 混淆矩阵 (测试模式)
- `error_samples.png`: 错误预测样本 (测试模式)

## 自定义使用

你可以在Python代码中直接使用各个组件：

```python
from src.models import MLP, CNN
from src.data_utils import get_mnist_datasets, get_data_loaders
from src.trainer import Trainer

# 加载数据
train_dataset, test_dataset = get_mnist_datasets('./data')
train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)

# 创建模型
model = MLP()  # 或 CNN()

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model, device, train_loader, test_loader)
trainer.train(epochs=10)
```

## 依赖项

- Python 3.7+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- matplotlib >= 3.3.0
- tqdm >= 4.62.0
- pillow >= 8.3.0
- scikit-learn (用于混淆矩阵)

## 许可证

MIT License