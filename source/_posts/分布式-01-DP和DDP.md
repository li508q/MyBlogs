---
title: 分布式-01-DP和DDP
date: 2024-09-15 12:00:00
updated: 2024-09-15 12:00:00
tags: 分布式训练
categories: 系统
---

> 首先 **DP** 和 **DDP** 都只是 `数据并行` 并不涉及到 `模型权重` 的拆分。

## DataParallel (DP)

DP是较简单的一种数据并行方式，直接将模型复制到多个GPU上并行计算，每个GPU计算batch中的一部分数据，各自完成前向和反向后，将梯度汇总到主GPU上。其基本流程：

1. 加载模型、数据至内存；
2. 创建DP模型；
3. DP模型的forward过程：
    1. **一个batch的数据均分到不同device**上；
    2. 为每个device复制一份模型；
    3. 至此，每个device上有模型和一份数据，并行进行前向传播；
    4. 收集各个device上的输出；
4. 每个device上的模型反向传播后，收集梯度到主device上，更新主device上的模型，将模型广播到其他device上；
5. 3-4循环。

在DP中，只有一个主进程，主进程下有多个线程，每个线程管理一个device的训练。因此，DP中内存中只存在一份数据，各个线程间是共享这份数据的。DP和Parameter Server的方式很像。

**Demo:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 假设我们有一个简单的数据集类
class SimpleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# 假设我们有一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# 假设我们有一些数据
n_sample = 100
n_dim = 10
batch_size = 10
X = torch.randn(n_sample, n_dim)
Y = torch.randint(0, 2, (n_sample, )).float()

dataset = SimpleDataset(X, Y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== 注意：刚创建的模型是在 cpu 上的 ===== #
device_ids = [0, 1, 2]
model = SimpleModel(n_dim).to(device_ids[0])
model = nn.DataParallel(model, device_ids=device_ids)


optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        
        loss = nn.BCELoss()(outputs, targets.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
```

其中最重要的一行便是：

```python
model = nn.DataParallel(model, device_ids=device_ids)
```

注意，模型的参数和缓冲区都要放在`device_ids[0]`上。在执行`forward`函数时，模型会被复制到各个GPU上，对模型的属性进行更新并不会产生效果，因为前向完后各个卡上的模型就被销毁了。**只有在`device_ids[0]`上对模型的参数或者buffer进行的更新才会生效！**

## DistributedDataParallel (DDP)

**DistributedDataParallel（DDP）** 是 PyTorch 提供的分布式数据并行训练接口，旨在高效地在多 GPU、甚至多机多 GPU 环境下进行训练。与 `DataParallel`（DP）相比，DDP 具有更高的效率和更好的可扩展性。

**DDP 的核心思想：**

- **多进程并行**：为每个 GPU 启动一个独立的进程，每个进程负责在其 GPU 上执行模型的前向和反向传播。
- **梯度同步**：在反向传播过程中，各进程之间通过通信（如 NCCL 后端）同步梯度，确保模型参数在所有进程中保持一致。
- **数据划分**：使用分布式采样器（`DistributedSampler`），确保每个进程处理的数据不重叠，实现数据并行。

### DDP 的执行流程
	
#### 1. **准备阶段**

##### a. **环境初始化**
- **初始化进程组**：使用 `torch.distributed.init_process_group`，指定通信后端（如 NCCL）、进程组名称等。
- **设置设备**：使用 `torch.cuda.set_device(local_rank)`，将当前进程绑定到指定的 GPU。

##### b. **模型广播**
- **创建模型实例**：在各个进程中创建模型实例，并将其移动到对应的 GPU 上。
- **封装 DDP 模型**：使用 `torch.nn.parallel.DistributedDataParallel` 封装模型。
- **模型参数广播**：DDP 会在后台自动将模型的参数和缓冲区从主进程广播到其他进程，确保模型初始状态一致。

##### c. **注册梯度钩子**
- **Reducer 管理器**：DDP 会为模型参数注册梯度钩子，在反向传播过程中自动进行梯度同步。

#### 2. **准备数据**
- **加载数据集**：使用标准的 PyTorch 数据集或自定义数据集。
- **创建分布式采样器**：使用 `torch.utils.data.distributed.DistributedSampler`，确保每个进程加载的数据不重叠。
- **创建数据加载器**：将采样器传递给数据加载器，以便在每个 epoch 开始时正确地划分数据。

#### 3. **训练阶段**
##### a. **前向传播**
- **模型前向计算**：每个进程使用其本地数据执行模型的前向传播。
- **同步参数和缓冲区**：在初始阶段，DDP 已经同步了参数和缓冲区。在训练过程中，缓冲区（如 BatchNorm 的 `running_mean` 和 `running_var`）的更新也会被自动同步。

##### b. **计算梯度**
- **反向传播**：每个进程独立计算梯度。
- **梯度同步**：DDP 在后台通过梯度钩子，使用异步的 All-Reduce 操作（如 NCCL）来平均梯度。
- **更新梯度状态**：当所有参数的梯度都被同步后，DDP 会将平均梯度写回参数的 `.grad` 属性。

##### c. **参数更新**
- **优化器更新参数**：使用优化器（如 SGD、Adam）更新模型参数。
- **参数一致性**：由于梯度已被同步，所有进程中的模型参数在更新后仍然保持一致。

#### 4. **循环训练**
- **重复上述步骤**，直到完成所有的训练迭代。

**Demo:**

```python
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 基础模块 ### 
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        cnt = torch.tensor(0)
        self.register_buffer('cnt', cnt)

    def forward(self, x):
        self.cnt += 1
        # print("In forward: ", self.cnt, "Rank: ", self.fc.weight.device)
        return torch.sigmoid(self.fc(x))

class SimpleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
# 2. 初始化我们的模型、数据、各种配置  ####
## DDP：从外部得到local_rank参数。从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

## DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

## 假设我们有一些数据
n_sample = 100
n_dim = 10
batch_size = 25
X = torch.randn(n_sample, n_dim)  # 100个样本，每个样本有10个特征
Y = torch.randint(0, 2, (n_sample, )).float()

dataset = SimpleDataset(X, Y)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

## 构造模型
model = SimpleModel(n_dim).to(local_rank)
## DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

## DDP: 构造DDP model —————— 必须在 init_process_group 之后才可以调用 DDP
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

## DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.BCELoss().to(local_rank)

# 3. 网络训练  ###
model.train()
num_epoch = 100
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    data_loader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in data_loader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label.unsqueeze(1))
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()

    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0 and epoch == num_epoch - 1:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
```

结合上面的代码，一个简化版的DDP流程：

1. 读取DDP相关的配置，其中最关键的就是：`local_rank`；
2. DDP后端初始化：`dist.init_process_group`；
3. 创建DDP模型，以及数据加载器。注意要为加载器创建分布式采样器（`DistributedSampler`）；
4. 训练。

DDP的通常启动方式：

```ini
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 ddp.py
```

### 一些概念

以上过程中涉及到一些陌生的概念，其实走一遍DDP的过程就会很好理解：每个进程是一个独立的训练流程，不同进程之间共享同一份数据。为了避免不同进程使用重复的数据训练，以及训练后同步梯度，进程间需要同步。因此，其中一个重点就是每个进程序号，或者说使用的GPU的序号。

- `node`：节点，可以是物理主机，也可以是容器；
- `rank`和`local_rank`：都表示进程在整个分布式任务中的编号。`rank`是进程在全局的编号，`local_rank`是进程在所在节点上的编号。显然，如果只有一个节点，那么二者是相等的。在启动脚本中的`--nproc_per_node`即指定一个节点上有多少进程；
- `world_size`：即整个分布式任务中进程的数量。

你好！你对 **DataParallel（DP）** 和 **DistributedDataParallel（DDP）** 的区别做了一个很好的总结。确实，DP 和 DDP 在实现方式、性能和适用场景上都有显著的不同。在分布式训练的实际应用中，涉及到很多复杂的细节，例如梯度的同步方式、数据采样策略以及进程间的通信等。

让我进一步深入探讨你提到的几个关键点，以帮助你更全面地理解 DP 和 DDP 的工作机制。

---


## DP 与 DDP 的区别

### **1. 并行方式**

- **DataParallel（DP）**：

  - **单进程多线程**：在一个进程中使用多线程实现并行计算。
  - **模型复制**：在每个前向传播中，将模型复制到多个 GPU 上。
  - **数据划分**：将输入数据划分成多个子批次，分别送入不同的 GPU。

- **DistributedDataParallel（DDP）**：

  - **多进程多线程**：为每个 GPU 启动一个独立的进程。
  - **进程间通信**：通过进程间通信（如 NCCL）来同步梯度和参数。
  - **更高的并行效率**：避免了 Python 全局解释器锁（GIL）的影响，提升了计算效率。

### **2. 性能差异的原因**

- **DP 通常比 DDP 慢，主要原因有：**

  1. **单进程的 GIL 限制**：DP 使用多线程并行计算，但由于 Python 的 GIL，无法真正实现并行计算，特别是在计算密集型任务中。

  2. **模型复制和数据划分的开销**：DP 在每次前向传播时都需要将模型复制到各个 GPU，并划分数据，这会增加额外的开销。

  3. **梯度汇总的瓶颈**：DP 在反向传播时需要将各个 GPU 的梯度汇总到主 GPU，这可能导致通信瓶颈。

- **DDP 的优势：**

  - **多进程并行，避免 GIL**：每个进程独立运行，GIL 不再成为瓶颈。

  - **高效的梯度同步**：使用 All-Reduce 操作，同步梯度更高效。

  - **通信开销更低**：DDP 支持 Ring-AllReduce，通信成本随着 GPU 数量的增加而 **相对固定**，而 DP 的通信成本则随着 GPU 数量线性增长。

### **3. 适用性**

- **DP 只能在单机上工作**，适用于小规模的多 GPU 训练。

- **DDP 可以在多机多卡上工作**，适用于大规模的分布式训练。

### **4. 模型并行的结合**

- **DDP 可以与模型并行相结合**：在需要模型并行的场景下，可以将模型的不同部分分配到不同的 GPU 上，同时使用 DDP 进行数据并行。

---

## **二、DP 与 DDP 中梯度的回收方式**

### **1. DP 中的梯度回收**

- **梯度计算**：在每个 GPU 上，模型副本计算其子批次数据的梯度。

- **梯度汇总**：所有 GPU 的梯度会被收集到主 GPU（`device_ids[0]`）上，进行汇总。

- **参数更新**：在主 GPU 上更新模型参数。

- **问题**：

  - **通信瓶颈**：所有梯度都需要传输到主 GPU，通信量大。

  - **主 GPU 的负载过重**：主 GPU 需要负责梯度汇总和参数更新，可能成为性能瓶颈。

### **2. DDP 中的梯度回收**

- **梯度计算**：每个进程独立计算其负责的数据的梯度。

- **梯度同步（All-Reduce 操作）**：

  - **All-Reduce**：将所有进程的梯度进行求和，然后平均分发回每个进程。

  - **异步通信**：DDP 采用异步的 All-Reduce 操作，可以与计算重叠，减少等待时间。

- **参数更新**：每个进程使用同步后的平均梯度，更新本地的模型参数。

- **优势**：

  - **通信效率高**：All-Reduce 操作的通信开销相对固定，不会随着 GPU 数量的增加而线性增长。

  - **没有单点瓶颈**：所有进程同时参与通信和计算，避免了主 GPU 的瓶颈。

### **3. 通信成本对比**

- **DP 的通信成本**：

  - 随着 GPU 数量的增加，通信成本 **线性增长**。

  - 主 GPU 需要收集和广播梯度，通信量大。

- **DDP 的通信成本**：

  - 使用 Ring-AllReduce，通信成本 **相对固定**。

  - 通信效率随着 GPU 数量的增加而 **更高效**。

---

## **三、DDP 中数据采样的细节**

### **1. 为什么需要 `DistributedSampler`**

- **数据划分的必要性**：在 DDP 中，每个进程都独立运行，为了避免不同进程处理相同的数据（数据重叠），需要确保每个进程处理的数据是互不重叠的子集。

- **`DistributedSampler` 的作用**：

  - **划分数据集**：将数据集划分为若干份，每个进程处理其中一份。

  - **确保随机性一致**：在每个 epoch 开始时，通过设置相同的随机种子，确保各进程的数据划分方式一致。

### **2. `DistributedSampler` 的工作机制**

- **分割数据集**：根据 `world_size`（总进程数）和 `rank`（当前进程编号），计算当前进程应该处理的数据索引范围。

- **处理数据不重叠**：不同进程处理的数据索引范围不重叠，确保了数据并行。

- **支持数据随机打乱**：在每个 epoch，可以通过设置不同的随机种子，实现数据的随机打乱。

### **3. 设置 `sampler.set_epoch(epoch)` 的必要性**

- **原因**：

  - **确保数据乱序的一致性**：在每个 epoch 开始时，需要为 `DistributedSampler` 设置 epoch，以确保所有进程的数据乱序方式一致。

  - **避免数据重复或遗漏**：不同进程在数据乱序时，如果不设置相同的种子，可能导致数据重复或遗漏，影响模型训练的正确性。

- **使用方法**：

  ```python
  data_loader.sampler.set_epoch(epoch)
  ```

---

## **四、DDP 中的数据同步操作**

### **1. 模型参数和缓冲区的同步**

- **初始同步**：

  - **参数广播**：在 DDP 初始化时，自动将主进程（`rank == 0`）的模型参数和缓冲区广播到其他进程，确保所有进程的模型状态一致。

- **缓冲区的同步**：

  - **自动同步**：在前向和反向传播过程中，DDP 会自动同步模型的缓冲区（如 BatchNorm 的 `running_mean` 和 `running_var`）。

  - **确保一致性**：使得模型在所有进程中的缓冲区状态保持一致。

### **2. 梯度的同步（All-Reduce 操作）**

- **注册梯度钩子**：DDP 为每个模型参数注册了梯度钩子，当参数的梯度计算完成后，自动触发 All-Reduce 操作。

- **All-Reduce 的过程**：

  - **梯度求和**：将所有进程的对应参数的梯度相加。

  - **梯度平均**：将总和除以进程数，得到平均梯度。

  - **同步更新**：将平均梯度分发回各个进程，更新模型参数。

### **3. 通信操作的处理**

- **通信后端**：通常使用高效的通信库（如 NCCL）进行进程间通信。

- **通信模式**：

  - **Broadcast（广播）**：用于初始参数和缓冲区的同步。

  - **All-Reduce**：用于梯度的同步。

  - **异步通信**：DDP 采用异步通信机制，通信和计算可以重叠，减少等待时间。

- **用户无需干预**：这些通信操作都由 DDP 在后台自动处理，用户不需要手动编写通信代码。

---

## **五、基于真实需求的实践体会**

- **复杂性与细节**：在分布式训练中，涉及到很多复杂的细节，包括通信机制、数据同步、随机性控制等。

- **实践的重要性**：只有在真实的项目中，面对具体的需求和挑战，才能深入理解并解决分布式训练中的各种问题。

- **建议**：

  - **深入学习 PyTorch 官方文档和示例**：了解 DDP 的详细使用方法和注意事项。

  - **从小规模实验开始**：先在单机多 GPU 环境下实践 DDP，熟悉其工作机制。

  - **逐步扩展到多机环境**：在熟悉基本原理后，可以尝试在多机多卡的环境下进行训练，处理更多的实际问题。

  - **关注性能优化**：在实践中，可以针对通信开销、数据加载效率、模型并行等方面进行优化，提升训练性能。

---

## **六、总结**

- **DP 与 DDP 的主要区别**在于并行方式、通信机制和性能表现。

- **DP 的局限性**：

  - 受限于 GIL，无法充分利用多核 CPU 和多 GPU 的计算能力。

  - 通信开销随着 GPU 数量线性增长，主 GPU 可能成为瓶颈。

- **DDP 的优势**：

  - 采用多进程并行，避开 GIL 限制，充分利用硬件资源。

  - 使用高效的 All-Reduce 操作，同步梯度和参数，通信开销低。

  - 支持多机多卡，具有良好的扩展性。

- **实践中需要注意的细节**：

  - 正确设置数据采样器，确保数据不重叠。

  - 理解梯度同步和参数更新的机制。

  - 熟悉 DDP 的启动和配置方法。

---

如果你还有其他疑问，或者希望深入了解某个具体的方面，例如 DDP 的启动方式、进程间通信的实现细节、模型并行的应用等，请随时告诉我！我很乐意继续为你解答。