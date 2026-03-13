import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    """
    模型训练器
    """
    def __init__(self, model, device, train_loader, test_loader, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 记录训练过程
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def test(self):
        """测试模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 累计损失
                test_loss += self.criterion(output, target).item()
                
                # 计算准确率
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        return test_loss, test_acc
    
    def train(self, num_epochs):
        """训练模型"""
        print(f"开始训练，总共 {num_epochs} 个epoch")
        print(f"使用设备: {self.device}")
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 测试
            test_loss, test_acc = self.test()
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), 'models/best_model.pth')
                print(f"保存最佳模型，准确率: {best_acc:.2f}%")
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        print(f'\n训练完成！最佳测试准确率: {best_acc:.2f}%')
        return best_acc
    
    def plot_history(self):
        """绘制训练历史"""
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.test_losses, 'r-', label='Test Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, self.test_accuracies, 'r-', label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()