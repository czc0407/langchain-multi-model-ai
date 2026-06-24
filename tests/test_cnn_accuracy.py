"""
验证 CNN 模块准确率 >= 98%
使用 FashionMNIST 测试集中类别 0,1,7,8 的样本
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from torchvision import datasets, transforms
from models.cnn_classifier import CNNClassifier
from config import CLASS_MAP

import tempfile

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
FASHION_DATA_DIR = os.path.join(tempfile.gettempdir(), "fashion_mnist_data")

print("=" * 60)
print("  CNN 准确率测试")
print("=" * 60)

print("\n[1/3] 加载 FashionMNIST 测试集...")
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = datasets.FashionMNIST(
    FASHION_DATA_DIR, train=False, download=True, transform=transform
)

selected_indices = [
    i for i in range(len(dataset))
    if dataset.targets[i].item() in [0, 1, 7, 8]
]
print(f"  测试集样本数（类别0,1,7,8）: {len(selected_indices)}")

print("\n[2/3] 加载 CNN 分类器...")
classifier = CNNClassifier()

print("\n[3/3] 逐样本预测...")
correct = 0
total = 0

for idx in selected_indices:
    img, label = dataset[idx]
    true_class = CLASS_MAP[label]  # 0,1,7,8 → 0,1,2,3

    img_np = img.numpy().reshape(1, 1, 28, 28)

    result = classifier.predict(img_np)
    pred_class = result["class_id"]

    if pred_class == true_class:
        correct += 1
    total += 1

    if total % 1000 == 0:
        print(f"  进度: {total}/{len(selected_indices)} "
              f"当前准确率: {correct/total*100:.2f}%")

accuracy = correct / total * 100
print(f"\n{'=' * 60}")
print(f"  CNN 准确率: {accuracy:.2f}% ({correct}/{total})")
print(f"  阈值: >= 98.00%")
if accuracy >= 98.0:
    print(f"  结果: PASS ✓")
else:
    print(f"  结果: FAIL ✗")
print(f"{'=' * 60}")

assert accuracy >= 98.0, f"CNN 准确率 {accuracy:.2f}% 未达标（需 >= 98%）"
