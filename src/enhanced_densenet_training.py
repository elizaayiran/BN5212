"""
增强版 DenseNet-121 CheXpert 模型
包含训练功能和Grad-CAM可解释性分析
Supporting Slide 5 & 6: Deep Learning Training + Explainable AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json

class CheXpertDataset(Dataset):
    """
    CheXpert数据集类
    支持多标签分类训练
    """
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # CheXpert 14个标签
        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 加载图像
        image_path = self.image_dir / f"{row['subject_id']}_{row['study_id']}.jpg"
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 提取标签 (将-1不确定标签转为0)
        labels = []
        for label in self.labels:
            value = row.get(label, 0)
            labels.append(1 if value == 1 else 0)  # 将-1和0都视为0
        
        return image, torch.FloatTensor(labels)

class DenseNetCheXpert(nn.Module):
    """
    基于DenseNet-121的CheXpert分类器
    Supporting Slide 5: Model Core - Building a Precise Diagnostic Engine
    """
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNetCheXpert, self).__init__()
        
        # 主干网络：DenseNet-121 (高参数效率，特征传播好)
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # 修改分类头
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        
        # 保存特征图用于Grad-CAM
        self.features = self.backbone.features
        self.classifier = self.backbone.classifier
        
        # 注册hook以获取特征图
        self.feature_maps = None
        self.gradients = None
        
    def forward(self, x):
        # 前向传播并保存特征图
        features = self.features(x)
        
        # 注册hook获取梯度
        if features.requires_grad:
            self.feature_maps = features
            features.register_hook(self.save_gradients)
        
        # 全局平均池化
        pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        pooled = torch.flatten(pooled, 1)
        
        # 分类
        output = self.classifier(pooled)
        return output
    
    def save_gradients(self, grad):
        """保存梯度用于Grad-CAM"""
        self.gradients = grad

class CheXpertTrainer:
    """
    CheXpert模型训练器
    Supporting Slide 5: Training Details
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # 损失函数：BCEWithLogitsLoss (适用于多标签任务)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 优化器：Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 计算准确率
                predictions = torch.sigmoid(output) > 0.5
                correct_predictions += (predictions == target.bool()).sum().item()
                total_predictions += target.numel()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        self.val_losses.append(avg_loss)
        self.scheduler.step(avg_loss)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10):
        """完整训练流程"""
        print("🚀 开始训练 CheXpert DenseNet-121 模型...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, 'best_chexpert_densenet121.pth')
                print("✅ 保存最佳模型")
        
        print("🎉 训练完成!")

class GradCAMExplainer:
    """
    Grad-CAM可解释性分析器
    Supporting Slide 6: Explainable AI (XAI)
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # CheXpert标签
        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    def generate_gradcam(self, image, class_idx):
        """
        生成Grad-CAM热力图
        打开"黑箱"：让AI的诊断看得懂
        """
        # 前向传播
        image = image.unsqueeze(0).to(self.device)
        image.requires_grad_()
        
        output = self.model(image)
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # 获取特征图和梯度
        feature_maps = self.model.feature_maps   # [1, 1024, 7, 7]
        gradients = self.model.gradients        # [1, 1024, 7, 7]
        
        # 计算权重 (全局平均池化梯度)
        weights = torch.mean(gradients, dim=(2, 3))  # [1, 1024]
        
        # 生成热力图
        cam = torch.zeros(
            feature_maps.shape[2:],
            dtype=feature_maps.dtype,
            device=feature_maps.device
        )
        for i in range(weights.shape[1]):
            cam += weights[0, i] * feature_maps[0, i]
        
        # ReLU激活
        cam = torch.relu(cam)
        
        # 归一化到[0,1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize_gradcam(self, image_path, class_idx, save_path=None):
        """
        可视化Grad-CAM结果
        将抽象的预测转化为直观的视觉证据
        """
        # 加载和预处理图像
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        original_image = Image.open(image_path).convert('RGB')
        input_image = transform(original_image)
        
        # 生成热力图
        cam = self.generate_gradcam(input_image, class_idx)
        
        # 调整热力图大小
        cam_resized = cv2.resize(cam, (224, 224))
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('原始胸部X光图像')
        axes[0].axis('off')
        
        # 热力图
        im1 = axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f'Grad-CAM: {self.labels[class_idx]}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # 叠加图像
        overlay = np.array(original_image.resize((224, 224)))
        cam_colored = plt.cm.jet(cam_resized)[:, :, :3]
        overlay_result = 0.6 * overlay/255.0 + 0.4 * cam_colored
        
        axes[2].imshow(overlay_result)
        axes[2].set_title('热力图叠加 (关注区域高亮)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Grad-CAM可视化保存到: {save_path}")
        
        plt.show()
        
        return cam_resized
    
    def explain_prediction(self, image_path, top_k=3):
        """
        全面解释模型预测
        生成多个标签的可解释性分析
        """
        # 预处理图像
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(Image.open(image_path).convert('RGB'))
        image_tensor = image.unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.sigmoid(output).cpu().numpy()[0]
        
        # 获取top-k预测
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        print("🔍 AI诊断可解释性分析")
        print("=" * 50)
        
        for i, idx in enumerate(top_indices):
            prob = probabilities[idx]
            label = self.labels[idx]
            
            print(f"\n{i+1}. {label}")
            print(f"   预测概率: {prob:.3f}")
            print(f"   置信度: {'高' if prob > 0.7 else '中' if prob > 0.3 else '低'}")
            
            # 生成并保存Grad-CAM
            save_path = f"gradcam_{label.replace(' ', '_').lower()}.png"
            self.visualize_gradcam(image_path, idx, save_path)
        
        return top_indices, probabilities[top_indices]

# 使用示例和训练脚本
def create_training_example():
    """创建训练示例"""
    print("📚 CheXpert DenseNet-121 训练示例")
    print("Supporting Slide 5: Deep Learning Model Training")
    
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集（需要实际的数据文件）
    # train_dataset = CheXpertDataset('train_labels.csv', 'train_images/', train_transform)
    # val_dataset = CheXpertDataset('val_labels.csv', 'val_images/', val_transform)
    
    # 创建数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseNetCheXpert(num_classes=14, pretrained=True)
    
    # 创建训练器
    trainer = CheXpertTrainer(model, device)
    
    print(f"✅ 模型结构: DenseNet-121")
    print(f"✅ 设备: {device}")
    print(f"✅ 损失函数: BCEWithLogitsLoss")
    print(f"✅ 优化器: Adam")
    
    # 开始训练（需要取消注释数据加载器）
    # trainer.train(train_loader, val_loader, epochs=20)

def create_explainability_example():
    """创建可解释性分析示例"""
    print("🔍 Grad-CAM可解释性分析示例")
    print("Supporting Slide 6: Explainable AI (XAI)")
    
    # 加载训练好的模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseNetCheXpert(num_classes=14, pretrained=True)
    
    # 加载权重（如果有的话）
    try:
        checkpoint = torch.load('best_chexpert_densenet121.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 加载训练好的权重")
    except:
        print("⚠️ 使用预训练权重（未针对CheXpert微调）")
    
    # 创建可解释性分析器
    explainer = GradCAMExplainer(model, device)
    
    print("🎯 Grad-CAM功能:")
    print("  - 生成热力图，显示模型关注的图像区域")
    print("  - 将抽象预测转化为直观视觉证据")
    print("  - 增强临床可信度和诊断透明度")
    
    # 示例用法（需要实际的图像文件）
    # explainer.explain_prediction('sample_chest_xray.jpg', top_k=3)

if __name__ == "__main__":
    print("🚀 CheXpert深度学习模型 - 训练与可解释性")
    print("=" * 60)
    
    # 创建训练示例
    create_training_example()
    
    print("\n" + "=" * 60)
    
    # 创建可解释性示例
    create_explainability_example()