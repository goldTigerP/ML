import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def predict_single_image(model, image_path, device, transform=None):
    """
    预测单张图像
    
    Args:
        model: 训练好的模型
        image_path: 图像路径
        device: 计算设备
        transform: 图像预处理变换
    
    Returns:
        predicted_class: 预测的类别
        confidence: 预测置信度
    """
    model.eval()
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    
    if transform:
        image = transform(image)
    else:
        # 默认变换
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        image = (image - 0.1307) / 0.3081  # 标准化
    
    # 添加batch维度
    image = image.unsqueeze(0).unsqueeze(0) if len(image.shape) == 2 else image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()

def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    可视化预测结果
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 显示的样本数量
    """
    model.eval()
    
    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 选择前num_samples个样本
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 预测
    images_gpu = images.to(device)
    with torch.no_grad():
        outputs = model(images_gpu)
        _, predictions = torch.max(outputs, 1)
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 显示图像
        img = images[i].squeeze().cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        
        # 设置标题
        pred = predictions[i].cpu().item()
        true_label = labels[i].item()
        color = 'green' if pred == true_label else 'red'
        axes[i].set_title(f'预测: {pred}\n真实: {true_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_errors(model, test_loader, device):
    """
    分析模型错误预测
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    error_samples = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 收集错误样本
            mask = predictions != labels
            if mask.any():
                error_indices = torch.where(mask)[0]
                for idx in error_indices:
                    error_samples.append({
                        'image': images[idx].cpu(),
                        'predicted': predictions[idx].cpu().item(),
                        'true_label': labels[idx].cpu().item()
                    })
    
    # 计算混淆矩阵
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels, all_predictions)
        use_sklearn = True
    except ImportError:
        print("警告: sklearn未安装，将使用简化的混淆矩阵")
        # 简单的混淆矩阵实现
        cm = np.zeros((10, 10), dtype=int)
        for true, pred in zip(all_labels, all_predictions):
            cm[true][pred] += 1
        use_sklearn = False
    
    if use_sklearn:
        print("分类报告:")
        print(classification_report(all_labels, all_predictions))
    else:
        # 计算简单的准确率统计
        print("分类统计:")
        for i in range(10):
            correct = cm[i, i]
            total = cm[i, :].sum()
            if total > 0:
                precision = correct / total
                print(f"数字 {i}: 准确率 {precision:.4f} ({correct}/{total})")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    
    # 在矩阵中添加数值
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 显示一些错误样本
    if error_samples:
        print(f"\n发现 {len(error_samples)} 个错误预测样本")
        
        # 显示前16个错误样本
        num_show = min(16, len(error_samples))
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_show):
            sample = error_samples[i]
            img = sample['image'].squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'预测: {sample["predicted"]}\n真实: {sample["true_label"]}', 
                            color='red')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_show, 16):
            axes[i].axis('off')
        
        plt.suptitle('错误预测样本', fontsize=16)
        plt.tight_layout()
        plt.savefig('error_samples.png', dpi=150, bbox_inches='tight')
        plt.show()

def model_summary(model, input_size=(1, 28, 28)):
    """
    打印模型摘要信息
    
    Args:
        model: 模型
        input_size: 输入尺寸
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            
            if hasattr(output, "size"):
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1
            else:
                summary[m_key]["output_shape"] = [[-1]]
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
        
        if (not isinstance(module, nn.Sequential) and 
            not isinstance(module, nn.ModuleList) and 
            not (module == model)):
            hooks.append(module.register_forward_hook(hook))
    
    summary = {}
    hooks = []
    
    model.apply(register_hook)
    
    # 创建虚拟输入
    x = torch.randn(1, *input_size)
    model(x)
    
    # 移除hooks
    for h in hooks:
        h.remove()
    
    print("="*70)
    print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
    print("="*70)
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        line_new = f"{layer:<25} {str(summary[layer]['output_shape']):<20} {summary[layer]['nb_params']:>15,}"
        total_params += summary[layer]["nb_params"]
        
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        
        print(line_new)
    
    print("="*70)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("="*70)