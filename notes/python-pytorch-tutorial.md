# Python → PyTorch 教程式笔记

配套代码：`src/day1_gpu_check.py`

这份笔记的读法不是“从头背到尾”，而是每章完成三件事：

1. 读懂正文里的心智模型。
2. 把示例代码亲手敲进 Python 跑一遍。
3. 做完章末练习，把结果写进 `log/`。

判断学会的标准不是“我看过了”，而是“我能不看示例写出来”。

---

## 第 1 章 Python：读 PyTorch 代码需要的最小语法

### 1.1 缩进就是代码块

Python 不用 `{}`，用缩进表示层级。同一个代码块必须缩进一致，通常用 4 个空格。

```python
def f(x):
    if x > 0:
        return x
    return -x
```

`if` 和 `return` 属于 `f`，因为它们比 `def` 多一层缩进。第二个 `return` 与 `if` 同级，所以它在 `if` 外面。

常见错误：混用 Tab 和空格。编辑器里统一设置成“Tab 转 4 空格”。

### 1.2 函数：输入、处理、返回

```python
def add(a, b=1):
    result = a + b
    return result
```

- `a` 是必填参数。
- `b=1` 是默认参数，不传时就是 1。
- `return` 同时做两件事：结束函数、把值交回调用处。

调用：

```python
add(3)        # 4
add(3, 5)     # 8
add(b=2, a=3) # 5，关键字传参
```

PyTorch 里大量关键字参数，例如 `nn.Conv2d(3, 64, kernel_size=3, padding=1)`。它的好处是不必死记参数顺序。

### 1.3 类：把数据和操作绑在一起

```python
class Counter:
    def __init__(self):
        self.value = 0

    def add(self):
        self.value += 1
        return self.value
```

- `__init__` 是构造函数，创建对象时自动执行。
- `self` 指当前对象，类似 Java/C++ 的 `this`，但 Python 要显式写出来。
- `self.value` 是对象属性，所有方法都能访问。

继承：

```python
class MyCounter(Counter):
    def __init__(self):
        super().__init__()
        self.name = "mine"
```

`MyCounter` 拥有 `Counter` 的方法。`super().__init__()` 先执行父类构造，避免父类该初始化的东西没初始化。

PyTorch 的 `class SimpleCNN(nn.Module)` 就是这个模式。

### 1.4 列表、元组、字典

```python
items = [1, 2, 3]              # 列表，可改
point = (3, 4)                 # 元组，通常不改
config = {"lr": 0.01, "epoch": 10}
```

取值：

```python
items[0]          # 1
point[1]          # 4
config["lr"]      # 0.01
config.get("bs", 32)  # 没有 bs 时返回 32，不报错
```

训练代码里经常出现：

```python
for images, labels in loader:
    ...
```

这表示每次从 `loader` 取出一个二元组，并自动拆成 `images` 和 `labels`。

### 1.5 切片：左闭右开

```python
a = [0, 1, 2, 3, 4, 5]
a[1:4]   # [1, 2, 3]
a[:3]    # [0, 1, 2]
a[3:]    # [3, 4, 5]
a[-1]    # 5
a[::-1]  # [5, 4, 3, 2, 1, 0]
```

最重要的一条：**包含起点，不包含终点**。所以 `a[1:4]` 的长度是 `4 - 1 = 3`。

NumPy、pandas、PyTorch 都沿用这套规则。

### 1.6 推导式：把循环压缩成一行

普通写法：

```python
squares = []
for x in range(5):
    squares.append(x * x)
```

推导式：

```python
squares = [x * x for x in range(5)]
```

读法：对 `range(5)` 里的每个 `x`，计算 `x * x`，收集成列表。

带条件：

```python
evens = [x for x in range(10) if x % 2 == 0]
```

不要嵌套太多层。超过一层 `for` 加一层 `if`，通常普通循环更清楚。

### 1.7 f-string：打印调试信息

```python
name = "model"
loss = 0.123456
print(f"{name} loss={loss:.4f}")
```

输出：

```text
model loss=0.1235
```

`:.4f` 表示保留 4 位小数。训练日志里大量使用这种格式。

### 1.8 with：自动清理资源

```python
with open("a.txt", encoding="utf-8") as f:
    text = f.read()
```

离开 `with` 后文件自动关闭。PyTorch 的：

```python
with torch.no_grad():
    pred = model(x)
```

也是同一思想：块内关闭梯度，离开后恢复。

### 1.9 `if __name__ == "__main__"`

直接运行文件时条件成立；被别的文件导入时不成立。它相当于把“可复用代码”和“直接运行入口”分开。

### 练习

1. 写一个函数 `mean3(a, b, c)` 返回三个数的平均值。
2. 写一个类 `Accumulator`，有 `add(x)` 和 `total()`，能累加数字。
3. 用推导式生成 0 到 20 中所有偶数的平方。

---

## 第 2 章 NumPy：先理解形状，再理解运算

### 2.1 ndarray 是什么

NumPy 数组可以理解为“一块规则的多维格子”。

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
print(a.shape)  # (3, 4)
print(a.ndim)   # 2
print(a.size)   # 12
print(a.dtype)
```

`reshape(3, 4)` 不改元素总数，只改排列方式：12 个元素变成 3 行 4 列。

读任何数组代码，先看 `shape`，不要先看值。

### 2.2 axis：沿着哪一维操作

```python
a = np.arange(12).reshape(3, 4)
a.sum(axis=0).shape  # (4,)
a.sum(axis=1).shape  # (3,)
```

`axis=0` 表示沿第 0 维求和，结果是第 0 维消失。

直觉：二维数组里，`axis=0` 是“把每一列的行加起来”，`axis=1` 是“把每一行的列加起来”。

三维时同理：

```python
b = np.arange(24).reshape(2, 3, 4)
b.sum(axis=0).shape  # (3, 4)
b.sum(axis=1).shape  # (2, 4)
b.sum(axis=2).shape  # (2, 3)
```

想保留被压缩的维度：

```python
a.sum(axis=1, keepdims=True).shape  # (3, 1)
```

保留长度 1 的维度，常常是为了下一步广播。

### 2.3 广播：形状不同也能运算

```python
x = np.arange(6).reshape(2, 3)
y = np.array([10, 20, 30])
print(x + y)
```

`y` 的形状是 `(3,)`，会被当成每一行都加 `[10, 20, 30]`。

规则从右往左对齐：

1. 维度不够，左边补 1。
2. 对应位置相等，可以运算。
3. 某一边是 1，可以拉伸。
4. 否则报错。

```python
z = np.array([[100], [200]])  # (2, 1)
print(x + z)
```

这里 `z` 被拉伸到每一列，实现“第一行加 100，第二行加 200”。

图像归一化 `x / 255.0` 也是广播：标量被应用到每个元素。

### 2.4 视图和副本

```python
a = np.arange(6)
b = a[1:4]
b[0] = 99
print(a)  # a 也被改了
```

基本切片通常返回视图，共享内存。想独立复制：

```python
b = a[1:4].copy()
```

这条规则在数据预处理里很重要：你以为只改了临时变量，结果把原数据也改了。

### 练习

1. 创建 `(2, 3, 4)` 数组，分别沿三个轴求和，打印结果形状。
2. 给 `(3, 4)` 数组的每一行分别加 `[1, 2, 3, 4]`。
3. 给 `(3, 4)` 数组的第一行加 10、第二行加 20、第三行加 30。

---

## 第 3 章 PyTorch Tensor：NumPy 加上 GPU 和梯度

### 3.1 张量的四个关键属性

```python
import torch

x = torch.randn(2, 3)
print(x.shape)
print(x.dtype)
print(x.device)
print(x.requires_grad)
```

- `shape`：形状。
- `dtype`：数据类型，模型计算通常用 `float32`。
- `device`：在 CPU 还是 GPU。
- `requires_grad`：是否记录梯度。

训练报错大多来自这四项不一致。

### 3.2 创建与转换

```python
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.zeros(2, 3)
z = torch.ones_like(x)
```

NumPy 转 Tensor：

```python
arr = np.array([1, 2, 3], dtype=np.float32)
t = torch.from_numpy(arr)
```

Tensor 转 NumPy：

```python
arr2 = t.numpy()
```

如果张量在 GPU 或需要梯度，要先处理：

```python
arr3 = t.detach().cpu().numpy()
```

分类标签通常要 `long`：

```python
labels = torch.tensor([0, 1, 2], dtype=torch.long)
```

### 3.3 形状操作

```python
x = torch.randn(8, 1, 28, 28)
x.reshape(8, -1).shape       # (8, 784)
x.flatten(1).shape           # (8, 784)
x.squeeze(1).shape           # (8, 28, 28)
x.unsqueeze(1).shape         # (8, 1, 1, 28, 28)
```

`-1` 表示“这一维自动推算”。

`view` 和 `reshape` 的区别：

- `view` 要求内存布局兼容，否则报错。
- `reshape` 能复用内存就复用，必要时复制。

因此日常优先用 `reshape` 或 `flatten`。

换轴：

```python
y = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
```

`permute` 后内存可能不连续，再接 `view` 要先：

```python
y = y.contiguous()
```

### 3.4 运算与归约

逐元素：

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
a + b
a * b
a ** 2
```

矩阵乘法：

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
(A @ B).shape  # (3, 5)
```

`*` 是逐元素乘，`@` 是矩阵乘，不要混用。

归约：

```python
a.sum(dim=1)
a.mean(dim=1)
a.max(dim=1).values
a.max(dim=1).indices
a.argmax(dim=1)
```

分类预测最常用：

```python
logits = torch.randn(8, 10)
pred = logits.argmax(dim=1)  # (8,)
```

### 练习

1. 创建 `(16, 3, 32, 32)` 张量，展平成 `(16, 3072)`。
2. 对 `(4, 5)` 张量沿第 1 维求最大值和 argmax。
3. 解释 `(8, 10)` 的 logits 为什么 `argmax(dim=1)` 后变成 `(8,)`。

---

## 第 4 章 autograd：梯度是怎么算出来的

### 4.1 计算图

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)
```

这里发生了三件事：

1. `requires_grad=True` 告诉 PyTorch 跟踪 `x`。
2. 后续运算形成计算图，`out` 知道它由哪些运算得到。
3. `backward()` 从 `out` 反向计算，把结果写入 `x.grad`。

这个例子的梯度是 4.5，可以手算验证。

只有标量能直接 `backward()`。非标量通常先 `mean()` 或 `sum()`，训练里的 `loss` 也是标量。

### 4.2 梯度会累加

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

如果下一轮不清梯度，新梯度会加到旧梯度上。因此标准顺序是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 4.3 推理时关闭梯度

```python
model.eval()
with torch.no_grad():
    pred = model(x)
```

- `eval()`：改变 Dropout、BatchNorm 的行为。
- `no_grad()`：不建计算图，省显存、提速。

两者作用不同，验证时通常一起用。

### 4.4 detach

```python
y = x.detach()
```

`y` 不再追踪梯度，但和 `x` 共享数据。常见于把中间结果当普通数值使用。

### 练习

1. 用 `x = torch.tensor(2.0, requires_grad=True)` 计算 `y = x ** 3`，求 `dy/dx`，结果应为 12。
2. 连续调用两次 `loss.backward()` 前不清梯度，观察 `.grad` 是否翻倍。

---

## 第 5 章 模型：nn.Module 和形状推导

### 5.1 定义模型的固定结构

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

规则：

1. 继承 `nn.Module`。
2. `super().__init__()` 必须先写。
3. 需要学习的层放到 `self` 上。
4. 前向逻辑写在 `forward`。
5. 调用时写 `model(x)`，不要写 `model.forward(x)`。

### 5.2 卷积和全连接

```python
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
linear = nn.Linear(128, 10)
```

`Conv2d` 输入输出都是 `(N, C, H, W)`。

卷积输出尺寸公式：

```text
out = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
```

最常用的 `kernel=3, stride=1, padding=1` 不改变高宽，只改通道数。

池化：

```python
pool = nn.MaxPool2d(2)
```

`2` 表示窗口 2×2，默认步长也是 2，高宽通常减半。

### 5.3 用 MNIST 走一遍形状

```python
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
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

输入 `(64, 1, 28, 28)`：

1. `conv1` 后 `(64, 32, 28, 28)`。
2. 池化后 `(64, 32, 14, 14)`。
3. `conv2` 后 `(64, 64, 14, 14)`。
4. 池化后 `(64, 64, 7, 7)`。
5. `flatten(1)` 后 `(64, 3136)`。
6. `fc1` 后 `(64, 128)`。
7. `fc2` 后 `(64, 10)`。

最后的 10 是 10 个类别的 logits，不是概率。

### 练习

1. 把输入改成 `(32, 3, 64, 64)`，设计两层卷积，最后输出 5 类，手算 `fc1` 输入维度。
2. 在 `forward` 每行后临时打印 `x.shape`，验证你的手算。

---

## 第 6 章 损失函数、优化器与 NaN

### 6.1 CrossEntropyLoss

多分类最常用：

```python
criterion = nn.CrossEntropyLoss()
logits = model(images)       # (N, C)
loss = criterion(logits, labels)  # labels: (N,), long
```

注意：

- `logits` 前不要手动 softmax，损失函数内部已经包含。
- `labels` 必须是整数类别，形状 `(N,)`，不是 `(N, 1)`。

预测：

```python
pred = logits.argmax(dim=1)
acc = (pred == labels).float().mean()
```

### 6.2 二分类

```python
criterion = nn.BCEWithLogitsLoss()
logits = model(x).squeeze(1)
loss = criterion(logits, y.float())
```

它内部包含 sigmoid，标签要 float。

### 6.3 优化器

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

优化器负责根据 `.grad` 更新参数。学习率太大可能发散，太小学得慢。

### 6.4 为什么归一化没做会 NaN

MNIST 原始像素是 0 到 255。直接输入网络，前向数值较大，配合较大学习率容易让参数和 loss 迅速变成 `inf/nan`。

处理：

```python
x = x.float() / 255.0
```

NaN 排查顺序：

1. 打印输入 `min/max`，确认是否归一化。
2. 降低学习率。
3. 检查标签是否越界或类型错误。
4. 用 `torch.isfinite(loss)` 检查。
5. 必要时用梯度裁剪。

### 练习

1. 用同一模型分别跑“不归一化”和“除以 255”，记录 loss。
2. 把学习率从 0.01 改成 0.001，比较一个 epoch 的 acc。

---

## 第 7 章 Dataset 与 DataLoader

### 7.1 Dataset 负责“取一条”

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

必须实现：

- `__len__`：总共有多少条。
- `__getitem__`：给下标，返回一条样本。

真实项目里，`__getitem__` 常常负责读图片、做增强、返回标签。

### 7.2 DataLoader 负责“组成一批”

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

它自动完成：

- 按 batch 取数据。
- 是否打乱。
- 把多条样本堆成批次张量。

```python
for images, labels in loader:
    print(images.shape, labels.shape)
    break
```

如果数据集有 42000 条、batch 64，则约有 657 个 batch。

### 7.3 从 CSV 到训练数据

```python
df = pd.read_csv(path)
x = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
x = x.reshape(-1, 1, 28, 28) / 255.0
y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
dataset = TensorDataset(x, y)
```

`df.iloc[:, 1:]` 跳过第 0 列标签，取所有像素列。

### 练习

1. 自己实现一个返回 100 条随机样本的 Dataset。
2. 用 batch size 16 遍历，打印每个 batch 的形状。

---

## 第 8 章 标准训练循环

### 8.1 训练五步

```python
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

口诀：前向、算损失、清梯度、反向、更新。工程上常把 `zero_grad` 放到最前面，也可以。

### 8.2 验证循环

```python
model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        val_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

val_loss /= total
val_acc = correct / total
```

乘以 `images.size(0)` 是为了按样本数加权平均，避免最后一个不满的 batch 影响均值。

### 8.3 GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)
labels = labels.to(device)
```

模型、输入、标签必须在同一设备。`.to()` 对张量不是原地操作，要接回返回值。

### 8.4 保存与加载

推荐只保存参数：

```python
torch.save(model.state_dict(), "best.pt")
```

加载：

```python
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()
```

训练中断续训还要保存 optimizer、epoch 等。

### 8.5 过拟合单 batch：最有效的调试法

正式训练前，先取一个 batch 重复训练 100 次：

```python
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

for _ in range(100):
    logits = model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

如果一个 batch 都学不下来，先查数据、标签、模型和学习率，不要直接跑全量。

---

## 第 9 章 实战：重写 Day 1 并做三组实验

目标：不看 `src/day1_gpu_check.py`，自己写出一个能运行的 MNIST 训练脚本。

要求：

1. 自己写 Dataset 加载、模型、训练循环。
2. 输入做归一化。
3. 模型和批次数据都放到 GPU。
4. 每轮打印 loss 和 acc。
5. 记录耗时和峰值显存。

三组实验，每次只改一个变量：

| 实验 | 改动 | 记录 |
|---|---|---|
| A | baseline：batch 64，SGD lr 0.01 | 时间、acc、显存 |
| B | batch 32 | 与 A 对比 |
| C | Adam lr 0.001 | 与 A 对比 |

写入 `exp/log.csv`：

```text
date,model,data,batch,optimizer,lr,epochs,time_sec,acc,peak_vram_mb,note
```

最后回答三个问题：

1. 为什么 GPU 上小模型只快了几倍？
2. 为什么 batch 改小后显存会下降？
3. 为什么 Adam 和 SGD 的合适学习率不同？

---

## 附录：报错速查

| 报错/现象 | 优先检查 |
|---|---|
| `Expected all tensors to be on the same device` | model/images/labels 是否都 `.to(device)` |
| `expected scalar type Long but found Float` | 分类标签是否 `.long()` |
| `view size is not compatible` | 改用 `reshape`，或先 `contiguous()` |
| loss 为 NaN | 输入归一化、学习率、标签范围 |
| CUDA out of memory | 减小 batch、检查是否保存了计算图、验证是否 `no_grad` |
| 验证结果忽高忽低 | 是否 `model.eval()` |
| 准确率一直等于随机猜测 | 标签对应关系、loss、学习率、是否忘记 `optimizer.step()` |
| 显存随训练持续上涨 | loss 是否 `.item()`，是否把带图张量存进列表 |

---

下一步主线仍然是水下检测，不要把这份笔记当成新的长期任务。遇到不懂的语法回来查对应章节即可。
