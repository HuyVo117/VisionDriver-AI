"""
VisionDriver AI — Training Pipeline cho Phase 2 (Driver Monitoring)
Fine-tune MobileNetV3-Small để nhận diện cảm xúc / trạng thái tài xế.

Classes: neutral, drowsy, panic, distracted, normal

Usage:
  python api/train_driver_state.py --data-dir data/driver_state/ --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small — nhẹ, phù hợp realtime inference."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Thay classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_transforms(train: bool = True):
    """Data augmentation cho training."""
    if train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def train(
    data_dir: str = "data/driver_state",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_path: str = "models/driver_state_model.pth",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"

    if not train_path.exists():
        print(f"[ERROR] Không tìm thấy: {train_path}")
        print("Cấu trúc cần có:")
        print("  data/driver_state/")
        print("  ├── train/")
        print("  │   ├── drowsy/     (ảnh tài xế buồn ngủ)")
        print("  │   ├── normal/     (ảnh tài xế tỉnh táo)")
        print("  │   ├── distracted/ (ảnh tài xế nhìn đi chỗ khác)")
        print("  │   └── panic/      (ảnh tài xế hoảng loạn)")
        print("  └── val/")
        return

    train_dataset = datasets.ImageFolder(str(train_path), transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(str(val_path), transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    print(f"[Train] Classes ({num_classes}): {train_dataset.classes}")

    model = get_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(
            f"Epoch [{epoch+1:3d}/{epochs}] "
            f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.classes,
            }, output_path)
            print(f"   ✅ Best model saved (val_acc={val_acc:.1f}%)")

    print(f"\nTraining done. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/driver_state")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/driver_state_model.pth")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
    )
