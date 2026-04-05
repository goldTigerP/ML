#!/usr/bin/env python3
"""
从已保存的 resnet18_cifar10.pth 权重文件导出 ONNX 模型（无需重新训练）
"""

import torch
import sys
import os

# 复用 resnet_train.py 中的 build_model 函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet_train import build_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 项目根目录 (mnist_recognition 的上一级)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 加载模型
    pth_path = os.path.join(models_dir, 'resnet18_cifar10.pth')
    if not os.path.exists(pth_path):
        print(f"错误: 找不到权重文件 {pth_path}")
        return

    model = build_model(pretrained=False, in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    print(f"已加载权重: {pth_path}")

    # 导出 ONNX
    # CIFAR-10 输入尺寸: batch=1, channels=3, height=32, width=32
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    onnx_path = os.path.join(models_dir, 'resnet18_cifar10.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
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
        print("提示: pip install onnx 可进行完整性验证")
    except Exception as e:
        print(f"ONNX 模型验证失败: {e}")

    # ONNX Runtime 一致性验证
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
            print("推理结果一致性验证通过 ✓")
        else:
            print("警告: 输出差异较大，请检查导出是否正确")
    except ImportError:
        print("提示: pip install onnxruntime 可验证推理一致性")
    except Exception as e:
        print(f"ONNX Runtime 验证失败: {e}")


if __name__ == '__main__':
    main()
