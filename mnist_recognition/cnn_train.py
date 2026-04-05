#!/usr/bin/env python3
"""
CNN 训练脚本（使用与 simple_train.py 相同的 MNIST 数据预处理及流程）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class SimpleCNN(nn.Module):
    """A small CNN for MNIST (single-channel 28x28 images).

    Architecture:
      conv(1->32,3) -> ReLU -> conv(32->64,3) -> ReLU -> MaxPool(2)
      -> Dropout2d(0.25) -> Flatten -> FC(64*12*12 -> 128) -> ReLU -> Dropout(0.5) -> FC(128->10)

    This is intentionally small so it trains quickly for experiments.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # after two convs (3x3, no padding) and one 2x2 pool on 28x28 input:
        # 28 -> conv1 -> 26 -> conv2 -> 24 -> pool -> 12
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def unnormalize(img_tensor, mean=0.1307, std=0.3081):
    # img_tensor: (C,H,W) or (1,H,W)
    return img_tensor * std + mean


def main():
    print("=== CNN MNIST 训练脚本 ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("下载 MNIST 数据集...")
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    train_losses = []
    train_accs = []

    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = output.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f'Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Acc = {epoch_acc:.2f}%')

    # 测试
    print("\n测试模型...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predicted = outputs.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_acc = 100 * correct / total
    print(f'测试准确率: {test_acc:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), './models/cnn_model.pth')
    print("模型已保存到 ./models/cnn_model.pth")

    # 导出 ONNX 格式模型
    print("\n导出 ONNX 模型...")
    model.eval()
    # MNIST 输入尺寸: batch=1, channels=1, height=28, width=28
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    onnx_path = './models/cnn_mnist.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,           # 将训练好的参数一起导出
        opset_version=13,             # ONNX 算子集版本
        do_constant_folding=True,     # 常量折叠优化
        input_names=['input'],        # 输入节点名称
        output_names=['output'],      # 输出节点名称
        dynamic_axes={                # 支持动态 batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )
    print(f"ONNX 模型已保存到 {onnx_path}")

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过 ✓")
    except ImportError:
        print("提示: 安装 onnx 包 (pip install onnx) 可对导出的模型进行完整性验证")
    except Exception as e:
        print(f"ONNX 模型验证失败: {e}")

    # 使用 ONNX Runtime 验证推理结果一致性
    try:
        import onnxruntime as ort
        import numpy as np

        ort_session = ort.InferenceSession(onnx_path)
        dummy_np = dummy_input.cpu().numpy()
        ort_outputs = ort_session.run(None, {'input': dummy_np})[0]

        with torch.no_grad():
            torch_outputs = model(dummy_input).cpu().numpy()

        max_diff = np.max(np.abs(torch_outputs - ort_outputs))
        print(f"PyTorch 与 ONNX Runtime 输出最大差异: {max_diff:.6e}")
        if max_diff < 1e-5:
            print("ONNX Runtime 推理结果一致性验证通过 ✓")
        else:
            print("警告: 输出差异较大，请检查导出是否正确")
    except ImportError:
        print("提示: 安装 onnxruntime (pip install onnxruntime) 可验证推理结果一致性")
    except Exception as e:
        print(f"ONNX Runtime 验证失败: {e}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存到 cnn_training_history.png")

    # 可视化预测结果（反归一化后显示）
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)  # images are normalized tensors

    with torch.no_grad():
        images_gpu = images[:10].to(device)
        outputs = model(images_gpu)
        predictions = outputs.argmax(dim=1)

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        img = unnormalize(images[i]).squeeze().numpy()  # back to ~[0,1]
        plt.imshow(img, cmap='gray')

        pred = predictions[i].cpu().item()
        true_label = labels[i].item()
        color = 'green' if pred == true_label else 'red'
        plt.title(f'预测: {pred}\n真实: {true_label}', color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('cnn_predictions.png', dpi=150, bbox_inches='tight')
    print("预测结果已保存到 cnn_predictions.png")

    print(f"\n训练完成！最终测试准确率: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
