"""实验脚本:改 BATCH_SIZE 和卷积通道数,观察速度与效果变化。

复用 day1_gpu_check.py 里的 load_dataset(感谢 if __name__ == "__main__" 保护,
import 时不会触发训练)。所有实验只跑 GPU、只跑 1 个 epoch、随机种子固定为 0,
保证互相可比。
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from day1_gpu_check import load_dataset


class TunableCNN(nn.Module):
    """和 SimpleCNN 一样,但通道数可以自己定。"""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def run(device, dataset, batch_size, c1, c2, tag, optimizer_name="sgd"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    torch.manual_seed(0)
    model = TunableCNN(c1, c2).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
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
        f"{tag:<16} | bs {batch_size:>3} | ch {c1:>2}/{c2:>2} | "
        f"{elapsed:6.2f}s | batches {len(loader):>4} | "
        f"loss {loss_sum / len(loader):.4f} | acc {correct / total:.4f}",
        flush=True,
    )
    return elapsed


if __name__ == "__main__":
    device = torch.device("cuda")
    dataset = load_dataset()
    print(f"images {len(dataset)}\n")

    # 实验1: batch size 的影响(模型不变,都是 32/64 通道)
    run(device, dataset, batch_size=32, c1=32, c2=64, tag="实验1: batch=32")
    run(device, dataset, batch_size=128, c1=32, c2=64, tag="实验1: batch=128")

    # 实验2: 通道数减半(batch 固定 64)
    run(device, dataset, batch_size=64, c1=16, c2=32, tag="实验2: 通道减半")

    # 实验3: 优化器对比(batch 64、通道 32/64,只换优化器)
    run(device, dataset, batch_size=64, c1=32, c2=64, tag="实验3: SGD")
    run(device, dataset, batch_size=64, c1=32, c2=64, tag="实验3: Adam", optimizer_name="adam")
