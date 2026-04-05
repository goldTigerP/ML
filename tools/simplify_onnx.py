#!/usr/bin/env python3
"""
ONNX Simplifier 模型简化脚本

演示如何使用 onnxsim 对 ONNX 模型进行图优化简化，包括：
  1. 常量折叠 (Constant Folding)
  2. 冗余算子消除 (Redundant Op Elimination)
  3. 算子融合 (Operator Fusion)

对比简化前后的：模型文件大小、节点数量、推理结果一致性
"""

import os
import sys
import onnx
import onnxsim
import numpy as np


def _get_total_model_size(model_path):
    """获取模型的总大小（包括外部数据文件）"""
    total = os.path.getsize(model_path)
    model_dir = os.path.dirname(model_path)
    basename = os.path.splitext(os.path.basename(model_path))[0]
    # 检查常见的外部数据文件
    for f in os.listdir(model_dir):
        if f.startswith(basename) and f.endswith('.onnx_data'):
            total += os.path.getsize(os.path.join(model_dir, f))
        # PyTorch dynamo 导出时会生成同名 .weight 文件
        if f == os.path.basename(model_path) + '.data':
            total += os.path.getsize(os.path.join(model_dir, f))
    return total


def get_model_info(model_path):
    """获取 ONNX 模型的基本信息"""
    model = onnx.load(model_path)
    graph = model.graph

    # 统计节点数
    num_nodes = len(graph.node)

    # 统计各类算子
    op_counts = {}
    for node in graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

    # 文件大小
    file_size = os.path.getsize(model_path)

    # 输入输出信息
    inputs = [(inp.name, [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim])
              for inp in graph.input]
    outputs = [(out.name, [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim])
               for out in graph.output]

    return {
        'num_nodes': num_nodes,
        'op_counts': op_counts,
        'file_size': file_size,
        'inputs': inputs,
        'outputs': outputs,
        'opset_version': model.opset_import[0].version if model.opset_import else 'unknown',
    }


def print_model_info(info, label="模型"):
    """打印模型信息"""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Opset 版本:   {info['opset_version']}")
    total_size = info.get('total_size', info['file_size'])
    print(f"  模型总大小:   {total_size / 1024:.1f} KB  (onnx文件: {info['file_size'] / 1024:.1f} KB)")
    print(f"  总节点数:     {info['num_nodes']}")
    print(f"  输入:         {info['inputs']}")
    print(f"  输出:         {info['outputs']}")
    print(f"  算子分布:")
    for op, count in sorted(info['op_counts'].items()):
        print(f"    {op:20s} x {count}")


def simplify_model(input_path, output_path):
    """
    使用 onnxsim 简化 ONNX 模型

    onnxsim 主要做以下优化:
      - 常量折叠: 将编译期可确定的计算提前算好，减少运行时开销
      - 冗余节点消除: 移除无用的 Identity、Reshape 等节点
      - 算子融合: 合并可以一起执行的相邻算子
      - Shape 推断: 补全所有中间张量的 shape 信息
    """
    print(f"\n加载模型: {input_path}")

    if not os.path.exists(input_path):
        print(f"  ❌ 文件不存在，跳过")
        return False

    # 加载原始模型（如果有外部数据，也一并加载）
    model = onnx.load(input_path, load_external_data=True)

    # 获取原始模型真实大小（包含外部数据文件）
    original_total_size = _get_total_model_size(input_path)

    # 打印简化前信息
    before_info = get_model_info(input_path)
    before_info['total_size'] = original_total_size
    print_model_info(before_info, f"简化前 - {os.path.basename(input_path)}")

    # === 核心: 使用 onnxsim.simplify() 简化 ===
    print(f"\n  正在简化...")
    try:
        simplified_model, check = onnxsim.simplify(model)

        if not check:
            print("  ⚠️  简化后模型验证未通过，使用原始模型")
            return False

        # 保存简化后的模型（权重内嵌，便于分发）
        onnx.save(simplified_model, output_path)
        print(f"  简化后模型已保存: {output_path}")

    except Exception as e:
        print(f"  ❌ 简化失败: {e}")
        return False

    # 打印简化后信息
    after_info = get_model_info(output_path)
    after_info['total_size'] = _get_total_model_size(output_path)
    print_model_info(after_info, f"简化后 - {os.path.basename(output_path)}")

    # === 对比报告 ===
    print(f"\n{'─'*50}")
    print(f"  📊 简化效果对比")
    print(f"{'─'*50}")

    # 使用 total_size 做公平对比（原始模型可能有外部数据文件）
    size_before = before_info.get('total_size', before_info['file_size'])
    size_after = after_info.get('total_size', after_info['file_size'])
    nodes_before = before_info['num_nodes']
    nodes_after = after_info['num_nodes']

    size_ratio = (1 - size_after / size_before) * 100 if size_before > 0 else 0
    node_ratio = (1 - nodes_after / nodes_before) * 100 if nodes_before > 0 else 0

    print(f"  模型大小:  {size_before/1024:.1f} KB → {size_after/1024:.1f} KB  ({size_ratio:+.1f}%)")
    print(f"  节点数量:  {nodes_before} → {nodes_after}  ({node_ratio:+.1f}%)")

    # 对比算子变化
    all_ops = set(list(before_info['op_counts'].keys()) + list(after_info['op_counts'].keys()))
    changed_ops = []
    for op in sorted(all_ops):
        b = before_info['op_counts'].get(op, 0)
        a = after_info['op_counts'].get(op, 0)
        if b != a:
            changed_ops.append((op, b, a))

    if changed_ops:
        print(f"\n  算子变化:")
        for op, b, a in changed_ops:
            print(f"    {op:20s}: {b} → {a}")
    else:
        print(f"\n  算子无变化（模型已经较优）")

    # === 推理一致性验证 ===
    try:
        import onnxruntime as ort

        print(f"\n  🔍 验证推理一致性...")

        # 获取输入 shape
        input_info = model.graph.input[0]
        input_shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                input_shape.append(dim.dim_value)
            else:
                input_shape.append(1)  # 动态维度用 1

        input_name = input_info.name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # 原始模型推理
        sess_before = ort.InferenceSession(input_path)
        out_before = sess_before.run(None, {input_name: dummy_input})[0]

        # 简化模型推理
        sess_after = ort.InferenceSession(output_path)
        out_after = sess_after.run(None, {input_name: dummy_input})[0]

        max_diff = np.max(np.abs(out_before - out_after))
        print(f"  最大输出差异: {max_diff:.6e}")
        if max_diff < 1e-5:
            print(f"  推理一致性验证通过 ✓")
        else:
            print(f"  ⚠️ 差异较大，请检查")

    except ImportError:
        print("\n  提示: pip install onnxruntime 可验证推理一致性")
    except Exception as e:
        print(f"\n  推理验证失败: {e}")

    return True


def main():
    # 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    print("=" * 60)
    print("  ONNX Simplifier 模型简化工具")
    print("=" * 60)

    # 定义要简化的模型列表: (输入路径, 输出路径, 描述)
    models = [
        ('cnn_mnist.onnx', 'cnn_mnist_simplified.onnx', 'SimpleCNN (MNIST)'),
        ('resnet18_cifar10.onnx', 'resnet18_cifar10_simplified.onnx', 'ResNet-18 (CIFAR-10)'),
    ]

    results = []
    for filename, out_filename, desc in models:
        input_path = os.path.join(models_dir, filename)
        output_path = os.path.join(models_dir, out_filename)

        print(f"\n\n{'#'*60}")
        print(f"  处理: {desc}")
        print(f"{'#'*60}")

        success = simplify_model(input_path, output_path)
        results.append((desc, success))

    # 汇总
    print(f"\n\n{'='*60}")
    print("  📋 汇总")
    print(f"{'='*60}")
    for desc, success in results:
        status = "✅ 完成" if success else "⏭️  跳过"
        print(f"  {desc:30s} {status}")

    # 列出 models 目录下所有 onnx 文件
    print(f"\n  models/ 目录下的 ONNX 文件:")
    for f in sorted(os.listdir(models_dir)):
        if f.endswith('.onnx'):
            size = os.path.getsize(os.path.join(models_dir, f))
            print(f"    {f:40s} {size/1024:8.1f} KB")


if __name__ == '__main__':
    main()
