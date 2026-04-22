"""Train an imitation learning model from collected CARLA data."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CarlaDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.images = sorted(self.data_dir.glob("frame_*.jpg"))

        self.labels = []
        with open(self.data_dir / "labels.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                self.labels.append(
                    {
                        "throttle": float(parts[1]),
                        "brake": float(parts[2]),
                        "steer": float(parts[3]),
                        "speed": float(parts[4]),
                    }
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0

        label = self.labels[idx]
        return torch.from_numpy(img).permute(2, 0, 1), {
            "throttle": torch.tensor(label["throttle"], dtype=torch.float32),
            "brake": torch.tensor(label["brake"], dtype=torch.float32),
            "steer": torch.tensor(label["steer"], dtype=torch.float32),
        }


class EndToEndModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.throttle_head = nn.Linear(64, 1)
        self.brake_head = nn.Linear(64, 1)
        self.steer_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        throttle = torch.sigmoid(self.throttle_head(x))
        brake = torch.sigmoid(self.brake_head(x))
        steer = torch.tanh(self.steer_head(x))
        return {"throttle": throttle, "brake": brake, "steer": steer}


def train(data_dir: str = "collected_data", epochs: int = 20, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CarlaDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EndToEndModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)

            predictions = model(images)

            loss = (
                criterion(predictions["throttle"], labels["throttle"].to(device).unsqueeze(1))
                + criterion(predictions["brake"], labels["brake"].to(device).unsqueeze(1))
                + criterion(predictions["steer"], labels["steer"].to(device).unsqueeze(1))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "end_to_end_model.pth")
    print("Model saved: end_to_end_model.pth")


if __name__ == "__main__":
    train()
