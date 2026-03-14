# ResNet-18（针对 MNIST 的适配版）

本文档说明在 `mnist_recognition/resnet_mnist_train.py` 中使用的 ResNet-18 模型的要点。
该脚本将标准 ResNet-18 适配为单通道 MNIST：用 3x3、stride=1 的 conv 替换原始的 conv1，并移除初始的 max-pool 层。下面的数值来自实例化该适配模型并对输入形状 (1,1,28,28) 进行一次前向传播后的测量。

- 总参数量：11,172,810

顶层模块与关键子模块的参数计数与输出形状（列出重要模块）：

1. conv1：Conv2d(1 -> 64, kernel=3, stride=1, padding=1)
   - 参数：576（权重 64*1*3*3 = 576；在 torchvision 的实现中通常不使用偏置）
   - 输出形状： (1, 64, 28, 28)
   - 每样本输出节点：64*28*28 = 50,176

2. bn1：BatchNorm2d(64)
   - 参数：128（scale 和 shift，通常记为 gamma 和 beta）
   - 输出形状： (1, 64, 28, 28)

3. layer1（包含两个 BasicBlock，通道数为 64）
   - 合计参数：147,968
   - 输出形状： (1, 64, 28, 28)
   - 每个 BasicBlock 包含两个 3x3 卷积和两个 BatchNorm 层

4. layer2（包含两个 BasicBlock，通道数为 128，第一块包含下采样 downsample）
   - 合计参数：525,568
   - 输出形状： (1, 128, 14, 14)
   - 第一块使用 1x1 的 downsample 卷积以匹配维度变化

5. layer3（包含两个 BasicBlock，通道数为 256）
   - 合计参数：2,099,712
   - 输出形状： (1, 256, 7, 7)

6. layer4（包含两个 BasicBlock，通道数为 512）
   - 合计参数：8,393,728
   - 输出形状： (1, 512, 4, 4)

7. fc（最终全连接层）
   - Linear(512 -> 10)
   - 参数：5,130（权重 512*10 = 5,120，偏置 10）
   - 输出形状： (1, 10)

说明：
- ResNet-18 由初始的 conv+bn 阶段及 4 个顺序的 "layer"（layer1..layer4）构成，每个 layer 包含两个残差 BasicBlock。当通道数翻倍时，部分 BasicBlock 会包含下采样分支（1x1 卷积）以匹配残差连接的维度。
- 参数的大头集中在较深的层（layer3 与 layer4），因为通道数（256、512）和卷积核数量都较大。
- 上述参数与输出形状来源于对 torchvision ResNet-18 的具体实例化（已按照小尺寸图像做了 conv1/maxpool 的适配），因此 conv1 的参数与 ImageNet 默认（7x7 卷积）不同，空间输出尺寸也相应变化。

如果你希望我把完整的按模块表（JSON 或 CSV）也导出到 `doc/`，我可以一并添加。

## FLOPs / MACs 分析（前向计算量）

- 定义说明：下面的计数以 MACs（multiply–accumulate，乘加次数）为单位；若将乘与加都计入 FLOPs，可近似视为 2 × MACs。

- 单样本前向总 MACs（输入 1x1x28x28）：455,800,832 MACs（约 455.8M MACs，约 911.6M FLOPs 若按乘与加分别计数）

- 主要计算项（按模块举例）：
   - conv1: 451,584 MACs
   - layer1 的每个 conv（如 layer1.0.conv1 / layer1.0.conv2 等）：约 28,901,376 MACs
   - layer2 的 conv（channel=128，空间 14x14）示例：约 14,450,688 / 28,901,376 MACs（分别对应 conv1 / conv2）
   - layer3 与 layer4 的 convs（channel=256/512，空间 7x7 / 4x4）也贡献了大量 MACs，构成总量的大头。

说明：对于 MNIST 这种小图像（28x28），虽然通道数较小，但 ResNet 的每层卷积仍在重复的空间上工作（因为为了保持分辨率我们移除了初始下采样），因此总体 MACs 依然很高（与采用标准 ImageNet 下采样的 ResNet-18 有显著差别）。

如果你需要，我可以把脚本 `tools/flops_output.json` 中的逐层明细复制到 `doc/`（文件名如 `doc/layer_summary_resnet_mnist.json`），以保存完整的逐层 MACs。
