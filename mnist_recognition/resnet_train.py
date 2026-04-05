
#!/usr/bin/env python3
"""
ResNet-18 训练脚本（用于 CIFAR-10）

- 使用 torchvision 的 ResNet-18（可选择 pretrained=False/True）
- 适配 CIFAR-10（3 通道、10 类），包含常见的数据增强与训练/测试流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def build_model(pretrained=False, in_channels=3, num_classes=10):
    """返回一个适配 in_channels 输入并修改分类头的 ResNet-18 模型。"""
    model = torchvision.models.resnet18(pretrained=pretrained)
    # For CIFAR-10 (32x32) it's common to replace the large 7x7,stride2 conv + maxpool
    # with a 3x3, stride1 conv and remove the initial maxpool to preserve spatial
    # resolution in early layers.
    if in_channels == 3:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # remove initial maxpool (not useful for small images)
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def unnormalize(img_tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """反归一化：接受 (C,H,W) Tensor，按通道还原到大致 [0,1] 范围并返回 numpy(H,W,C)。"""
    img = img_tensor.clone()
    if isinstance(mean, (float, int)):
        mean = (mean,) * img.shape[0]
    if isinstance(std, (float, int)):
        std = (std,) * img.shape[0]
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).cpu().numpy()


def main():
    print("=== ResNet-18 CIFAR-10 训练脚本 ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    print("下载 CIFAR-10 数据集...")
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    # DataLoader performance settings
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

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    model = build_model(pretrained=False, in_channels=3, num_classes=10).to(device)
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
    torch.save(model.state_dict(), './models/resnet18_cifar10.pth')
    print("模型已保存到 ./models/resnet18_cifar10.pth")

    # 导出 ONNX 格式模型
    print("\n导出 ONNX 模型...")
    model.eval()
    # CIFAR-10 输入尺寸: batch=1, channels=3, height=32, width=32
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    onnx_path = './models/resnet18_cifar10.onnx'
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
        # 用同一个 dummy_input 对比 PyTorch 和 ONNX Runtime 的输出
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
    plt.savefig('resnet_training_history_cifar10.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存到 resnet_training_history_cifar10.png")

    # 可视化预测结果（反归一化后显示） - 适用于 CIFAR RGB
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        images_gpu = images[:10].to(device)
        outputs = model(images_gpu)
        predictions = outputs.argmax(dim=1)

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        img = unnormalize(images[i], mean=cifar_mean, std=cifar_std)
        plt.imshow(img)

        pred = predictions[i].cpu().item()
        true_label = labels[i].item()
        color = 'green' if pred == true_label else 'red'
        plt.title(f'预测: {pred}\n真实: {true_label}', color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('resnet_predictions_cifar10.png', dpi=150, bbox_inches='tight')
    print("预测结果已保存到 resnet_predictions_cifar10.png")

    print(f"\n训练完成！最终测试准确率: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
