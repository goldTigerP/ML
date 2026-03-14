#!/usr/bin/env python3
"""
ResNet-18 训练脚本（用于 MNIST）

- 适配单通道 MNIST（28x28），将 ResNet-18 的 conv1 替换为 3x3 stride=1，移除初始 maxpool
- 使用常见的训练/测试流程，并启用 DataLoader 性能设置与 non_blocking 数据搬移
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def build_model(pretrained=False, in_channels=1, num_classes=10):
    model = torchvision.models.resnet18(pretrained=pretrained)
    # For small single-channel images (MNIST 28x28) use 3x3 stride1 conv and remove maxpool
    if in_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif in_channels == 3:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def unnormalize(img_tensor, mean=0.1307, std=0.3081):
    """反归一化：接受 (C,H,W) Tensor，按通道还原到大致 [0,1] 并返回 numpy(H,W) 或 (H,W,C)。"""
    img = img_tensor.clone()
    if img.dim() == 3 and img.shape[0] == 1:
        img = img[0]
    img = img * std + mean
    img = img.clamp(0.0, 1.0)
    return img.cpu().numpy()


def main():
    print("=== ResNet-18 MNIST 训练脚本 ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    mnist_mean = 0.1307
    mnist_std = 0.3081

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,)),
    ])

    print("下载 MNIST 数据集（如果尚未下载）...")
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=test_transform)

    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    num_workers = min(4, max(1, cpu_count - 1))

    train_batch_size = 128
    test_batch_size = 256

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

    model = build_model(pretrained=False, in_channels=1, num_classes=10).to(device)
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
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

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
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            outputs = model(data)
            predicted = outputs.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_acc = 100 * correct / total
    print(f'测试准确率: {test_acc:.2f}%')

    torch.save(model.state_dict(), './models/resnet18_mnist.pth')
    print("模型已保存到 ./models/resnet18_mnist.pth")

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
    plt.savefig('resnet_mnist_training_history.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存到 resnet_mnist_training_history.png")

    # 可视化预测结果（反归一化后显示）
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        images_gpu = images[:10].to(device, non_blocking=True)
        outputs = model(images_gpu)
        predictions = outputs.argmax(dim=1)

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        img = unnormalize(images[i], mean=mnist_mean, std=mnist_std)
        plt.imshow(img, cmap='gray')

        pred = predictions[i].cpu().item()
        true_label = labels[i].item()
        color = 'green' if pred == true_label else 'red'
        plt.title(f'预测: {pred}\n真实: {true_label}', color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('resnet_mnist_predictions.png', dpi=150, bbox_inches='tight')
    print("预测结果已保存到 resnet_mnist_predictions.png")

    print(f"\n训练完成！最终测试准确率: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
