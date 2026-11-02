import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA = r"C:\Users\20555\Documents\deeplearning\PYTORCH\kaggle\minis\train.csv"
BATCH_SIZE = 64


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_dataset():
    df = pd.read_csv(DATA)
    x = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
    x = x / 255.0
    y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
    return TensorDataset(x, y)


def train_one_epoch(device, loader):
    torch.manual_seed(0)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    start = time.perf_counter()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    elapsed = time.perf_counter() - start
    print(
        f"{device.type:>4} | {elapsed:6.2f}s | "
        f"loss {loss_sum / len(loader):.4f} | acc {correct / total:.4f}"
    )
    return elapsed


if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"images {len(dataset)} | batches {len(loader)} | batch_size {BATCH_SIZE}\n")

    cpu_time = train_one_epoch(torch.device("cpu"), loader)

    if torch.cuda.is_available():
        gpu_time = train_one_epoch(torch.device("cuda"), loader)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\npeak VRAM {peak_mb:.0f} MB | speedup {cpu_time / gpu_time:.1f}x")
