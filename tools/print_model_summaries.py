"""
Print model architectures and parameter counts using torchsummary.

Usage:
    python tools/print_model_summaries.py

The script will try to import the models from the repository and call
`torchsummary.summary` for each. If `torchsummary` is not installed,
it will print an instruction to install it.
"""
import importlib
import sys
import os
import traceback

import torch


def try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        print(f"无法导入模块 {name}:\n", traceback.format_exc())
        return None


def main():
    try:
        from torchsummary import summary
    except Exception:
        print("\n错误: 未安装 `torchsummary`。请运行: pip install torchsummary\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Ensure project root is on sys.path so local package imports work when running
    # the script from other directories or from an IDE.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # (name, module_path, attr_name, input_size, kwargs)
    models = [
        ("SimpleMLP", "mnist_recognition.simple_train", "SimpleMLP", (1, 28, 28), {}),
        ("SimpleCNN", "mnist_recognition.cnn_train", "SimpleCNN", (1, 28, 28), {}),
        ("ResNet18_MNIST", "mnist_recognition.resnet_mnist_train", "build_model", (1, 28, 28), {"in_channels": 1, "num_classes": 10}),
        ("ResNet18_CIFAR", "mnist_recognition.resnet_train", "build_model", (3, 32, 32), {"in_channels": 3, "num_classes": 10}),
    ]

    for name, module_path, attr_name, input_size, kwargs in models:
        print("=" * 80)
        print(f"Model: {name}")
        mod = try_import(module_path)
        if mod is None:
            print(f"跳过 {name}（无法导入模块 {module_path}）\n")
            continue

        if not hasattr(mod, attr_name):
            print(f"模块 {module_path} 中没有属性 {attr_name}，跳过。\n")
            continue

        attr = getattr(mod, attr_name)

        try:
            # If attr is a class, instantiate without args.
            if isinstance(attr, type):
                model = attr()
            else:
                # assume a builder function like build_model(pretrained=False, ...)
                try:
                    model = attr(pretrained=False, **kwargs)
                except TypeError:
                    # fallback: call without pretrained
                    model = attr(**kwargs)

            model = model.to(device)

            # torchsummary expects the device string
            dev_str = "cuda" if device.type == "cuda" else "cpu"
            print(f"Input size for summary: {input_size}, device={dev_str}")
            summary(model, input_size, device=dev_str)
        except Exception:
            print(f"打印 {name} 结构时出错:\n", traceback.format_exc())


if __name__ == "__main__":
    main()
