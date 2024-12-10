---
title: Huggingface-基础部分-07-模型保存和加载
date: 2024-12-10 11:00:00
cover: /img/huggingface.png
top_img: /img/huggingface_long.png
tags: Huggingface
categories: 算法
---

## 从 `torch` 开始

1. 模型结构
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 模型结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 输入大小为10，输出大小为20
        self.fc2 = nn.Linear(20, 1)   # 输入大小为20，输出大小为1

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        x = self.fc2(x)
        return x
```

2. 训练
```python
# 定义损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

# 假设我们有一些输入数据X和对应的目标值y
# 这里简单地用随机数生成示例数据
X = torch.randn(100, 10)  # 100个样本，每个样本10维特征
y = torch.randn(100, 1)   # 对应的目标值

# 数据需要转换为PyTorch的张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 开始训练！
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    outputs = model(X)  # 前向传播
    loss = criterion(outputs, y)  # 计算损失

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

3. 权重保存：torch提供了两种方式进行保存：
	- **保存整个模型**：保存整个模型的结构（代码）、参数。
	- **保存模型参数**：仅保存模型的参数，而不保存模型的结构（代码）。

方法一：保存整个模型的结构（代码）和参数：

```python
# 保存模型
torch.save(model, 'model.pth')
```

那如何使用呢？特别简单：

```python
# 加载整个模型
loaded_model = torch.load('model.pth')
# 直接进行推理
output = loaded_model(input_tensor)
```

方法二：只保存模型的参数，不保存模型的结构（代码）：

```python
# 保存模型参数
torch.save(model.state_dict(), 'model_params.pth')
```

使用和第一种方式有很大的差别：
	**要先实例化模型，也是说要有模型结构的代码，才能加载参数**：

```python
# 模型结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 输入大小为10，输出大小为20
        self.fc2 = nn.Linear(20, 1)   # 输入大小为20，输出大小为1

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        x = self.fc2(x)
        return x

# 加载模型参数
model = SimpleNN()  # 创建模型实例
model.load_state_dict(torch.load('model_params.pth'))
# 直接进行推理
output = model(input_tensor)
```

**小结：**

第一种方式(使用 `torch.save(model, 'model.pth')` 保存整个模型), 其实是在保存模型的时候，序列化的数据被绑定到了特定的类（代码中的模型类）和确切的目录，本质上是不保存模型结构（代码）本身，而是**保存这个模型结构（代码）的路径**，并且在加载的时候会使用，因此当在其他项目里使用或者重构的时候，这种方式加载模型的时候会出错。
	
这意味着：
	模型类的实际代码不会被保存在 `.pth` 文件中。
	仅保存了模型类及其模块路径的引用。
	加载模型时，Python 需要访问相同的类定义，且路径必须与保存时一致。

因此，如果移动了代码、重构了项目，或尝试在不同的环境中加载模型，而该环境中类定义不在预期的位置，你可能会遇到错误。因此推荐使用第二种方法（仅保存模型参数），这种方法更稳健且具有更好的移植性。

**PS**:
	这里再解释一下.pth和.bin的文件格式，两者都是 `二进制` 的格式，一个是torch保存的格式，一个是huggingface的保存格式。

## Huggingface

### 保存方式
Huggingface所保存的 `bin文件` 保存的是模型的参数，使用的是上述torch的第二种权重保存方式。因此，想要完整加载模型是需要模型结构（代码）的。

### 模型结构代码在哪里？

接上面的内容，既然我们需要模型结构的代码，那么加载的时候，这个代码在哪里？
答案是在huggingface这个包的代码里，以GPT2举例，看他的源码就能找到模型代码：

```python
"""PyTorch OpenAI GPT-2 model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 中间省略

class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

# 后面省略
```

所以使用huggingface时，都是：

```python
from transformers import AutoTokenizer, GPT2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

# 模型保存，格式为.bin，只保存参数
model.save_pretrained("MyGPT2")
```

### 如何修改模型？

到这里应该前面两个问题应该清晰了，接下来第三个问题，想修改这个模型需要怎么办？
答案是和torch一样，修改模型结构的代码，比如想要魔改GPT2：

```python
# 前面省略

class MyGPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 开始魔改！！！

# 后面省略
```

后面加载的时候就变成了

```python
from xx import MyGPT2Model

model = MyGPT2Model.from_pretrained("gpt2")
```

**注意**
 - 如果改的参数量和原参数量不一致了，还使用这种加载方式会出现问题。
 - 魔改要**注意很多地方的兼容性**，一般魔改后是为了从头预训练的。


因而，huggingface简单来说他就是帮我们实现了各个模型结构的代码。
- 这样从网上下载模型的参数权重，直接加载就能够使用。
- 所以不用huggingface，只用torch也是可以加载各种模型权重的，只是需要自己实现模型结构代码，比如纯torch实现一个超简单版的的GPT2：

```python
import torch
import torch.nn as nn

class SimpleGPT2Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleGPT2Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_decoder = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(self.transformer_decoder, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer(x)
        return output

# 创建简化的 GPT-2 解码器模型
vocab_size = 10000  # 假设词汇表大小为10000
d_model = 256        # 假设嵌入维度为256
nhead = 8            # Transformer中的头数
num_layers = 6       # Transformer的层数
simple_gpt2 = SimpleGPT2Decoder(vocab_size, d_model, nhead, num_layers)

# 保存模型参数
torch.save(simple_gpt2.state_dict(), 'simple_gpt2.pth')

# 加载模型参数
loaded_model = SimpleGPT2Decoder(vocab_size, d_model, nhead, num_layers)
loaded_model.load_state_dict(torch.load('simple_gpt2.pth'))

# 进行推理
input_tensor = torch.tensor([[1, 2, 3, 4, 5]])  # 输入数据，假设为长度为5的序列
output = loaded_model(input_tensor)
print(output)
```