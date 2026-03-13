#!/usr/bin/env python3
"""
MNIST手写数字识别 - 主程序

使用PyTorch实现MLP和CNN模型进行MNIST手写数字识别
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import MLP, CNN
from data_utils import get_mnist_datasets, get_data_loaders
from trainer import Trainer
from utils import visualize_predictions, analyze_errors, model_summary

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MNIST手写数字识别')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help='选择模型类型 (mlp 或 cnn)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='禁用CUDA')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='运行模式: train(训练), test(测试), demo(演示)')
    
    args = parser.parse_args()
    
    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 加载数据
    print("加载MNIST数据集...")
    train_dataset, test_dataset = get_mnist_datasets('./data')
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, args.batch_size)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 创建模型
    if args.model == 'mlp':
        model = MLP()
        model_name = "MLP"
    else:
        model = CNN()
        model_name = "CNN"
    
    print(f"\n创建 {model_name} 模型:")
    model_summary(model)
    
    if args.mode == 'train':
        # 训练模式
        print(f"\n开始训练 {model_name} 模型...")
        trainer = Trainer(model, device, train_loader, test_loader, args.lr)
        best_acc = trainer.train(args.epochs)
        
        # 绘制训练历史
        trainer.plot_history()
        
        # 可视化预测结果
        print("\n生成预测可视化...")
        visualize_predictions(model, test_loader, device)
        
        print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")
    
    elif args.mode == 'test':
        # 测试模式
        model_path = f'models/best_model.pth'
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在！请先训练模型。")
            return
        
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 测试模型
        trainer = Trainer(model, device, train_loader, test_loader)
        test_loss, test_acc = trainer.test()
        
        print(f"\n测试结果:")
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.2f}%")
        
        # 错误分析
        print("\n进行错误分析...")
        analyze_errors(model, test_loader, device)
    
    elif args.mode == 'demo':
        # 演示模式
        model_path = f'models/best_model.pth'
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在！请先训练模型。")
            return
        
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        print("\n演示预测结果...")
        visualize_predictions(model, test_loader, device, num_samples=10)

def quick_start():
    """
    快速开始函数 - 使用默认参数训练MLP模型
    """
    print("=== MNIST手写数字识别 - 快速开始 ===")
    print("使用默认参数训练MLP模型...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 加载数据
    print("加载MNIST数据集...")
    train_dataset, test_dataset = get_mnist_datasets('./data')
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size=64)
    
    # 创建和训练MLP模型
    model = MLP()
    print("\n创建MLP模型:")
    model_summary(model)
    
    print("\n开始训练...")
    trainer = Trainer(model, device, train_loader, test_loader)
    best_acc = trainer.train(10)  # 训练10个epoch
    
    # 可视化结果
    trainer.plot_history()
    visualize_predictions(model, test_loader, device)
    
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 如果没有提供命令行参数，运行快速开始
        quick_start()
    else:
        # 否则使用命令行参数
        main()