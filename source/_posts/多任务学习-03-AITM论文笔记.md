---
title: 多任务学习-03-AITM论文笔记
---

## 重点：写在前面
```python
# AIT Module - 自适应信息传递模块
# 注意这里与Transformer中的Self-Attention不同的是，我只需要q_t作为query的最终输出即可
# 所以这里使用的就是最一般的Attention
# Q: [B, n, d] -> [B, 1, d]
# K: [B, m, d] -> [B, 2, d]
# V: [B, m, v] -> [B, 2, d]
class AITModule(nn.Module):
    def __init__(self, input_dim):
        super(AITModule, self).__init__()
        self.h1 = nn.Linear(input_dim, input_dim)
        self.h2 = nn.Linear(input_dim, input_dim)
        self.h3 = nn.Linear(input_dim, input_dim)
        self.w_c = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, p_t_minus_1, q_t):
        """
            p_t_minus_1: [B, dim]
            q_t: [B, dim]
        """
        p_t_minus_1 = p_t_minus_1.unsqueeze(1)  # [B, 1, dim]
        q_t = q_t.unsqueeze(1)  # [B, 1, dim]
        inputs = torch.cat([p_t_minus_1, q_t], dim=1)  # [B, 2, dim]
        
        q = self.h1(q_t)  # [B, 1, d]
        k = self.h2(inputs)  # [B, 2, d]
        v = self.h3(inputs)  # [B, 2, d]

        score = q @ k.permute(0, 2, 1) / q.size(-1) ** 0.5  # [B, 1, 2]
        weight = self.softmax(score)  # [B, 1, 2]

        weighted_v = weight @ v  # [B, 1, dim]
        output = self.w_c(weighted_v.squeeze(1))  # [B, dim]
        return output
```

## 1. 问题表述

### 多步骤转化问题

在广告和推荐系统中，用户的最终购买行为通常是一个多步骤的过程。例如，用户可能首先点击广告（点击率，CTR），然后在商品页面浏览（浏览率，View Rate），再将商品加入购物车（加入购物车率，Add-to-Cart Rate），最后完成购买（购买率，Purchase Rate）。这些步骤之间存在**序列依赖关系**，即前一个行为的发生是后一个行为发生的前提条件。

### 序列依赖的标签

对于每个用户和广告的交互，我们有一个输入特征向量 $\mathbf{x}$，以及多个转化步骤的标签 $y_t$ ，其中 $t = 1, 2, \ldots, T$ 。标签 $y_t$ 表示用户是否在步骤 $t$ 完成了对应的行为（1 表示完成，0 表示未完成）。

由于序列依赖关系，我们有 $y_1 \geq y_2 \geq \cdots \geq y_T$ 。这意味着，如果用户没有完成步骤 $t$，则后续的步骤 $t+1, t+2, \ldots, T$ 也不可能完成。

### 目标

我们的目标是基于输入特征 $\mathbf{x}$ 预测每个步骤 $t$ 的**端到端转化概率** $\hat{y}_t$，即用户从第一步到第 $t$ 步都完成的概率：

$$
\hat{y}_t = p(y_1 = 1, y_2 = 1, \ldots, y_t = 1 | \mathbf{x})。
$$

---

## 2. 自适应信息传递多任务（AITM）框架

### 共享嵌入层

- **嵌入层**：将离散的输入特征 $x_i$ 映射为低维密集向量 $\mathbf{v}_i \in \mathbb{R}^d$ ，捕获特征的潜在表示。
- **特征拼接**：将所有嵌入向量拼接为一个大的特征向量 $\mathbf{v}$，作为后续模型的输入。

**共享嵌入的优势**：

1. **信息共享**：前期任务（如点击）通常有更多的正样本，模型可以通过共享嵌入从这些任务中学习有用的特征，有助于后续任务的学习。
2. **参数减少**：共享嵌入层避免了为每个任务分别学习嵌入，减少了模型的参数量，提高了训练效率。

### 任务塔（Task Tower）

- **定义**：每个任务 $t$ 都有一个独立的塔结构 $f_t(\cdot)$ ，用于从共享的输入特征 $\mathbf{v}$ 中学习任务特定的表示 $\mathbf{q}_t$ 。
- **灵活性**：塔结构可以是任何先进的模型，如 NFM、DeepFM、AFM 等，这使得 AITM 框架具有高度的通用性。

### 自适应信息传递模块（AIT 模块）

**目的**：在任务之间建模序列依赖关系，允许信息从前一个任务自适应地传递到当前任务。

#### 信息传递机制

1. **获取前一个任务的信息**：

$$
   \mathbf{p}_{t-1} = g_{t-1}(\mathbf{z}_{t-1})
$$

   - $\mathbf{z}_{t-1}$ 是前一个任务经过 AIT 模块后的输出。
   - $g_{t-1}(\cdot)$ 是一个函数，用于从 $\mathbf{z}_{t-1}$ 中提取需要传递的信息(Info: 信息桥)。

2. **自适应融合信息**：

$$
   \mathbf{z}_t = \sum_{u \in \{\mathbf{p}_{t-1}, \mathbf{q}_t\}} w_u h_1(u)
$$

   - **输入**：前一个任务的传递信息 $\mathbf{p}_{t-1}$ 和当前任务的原始表示 $\mathbf{q}_t$ 。
   - **权重$w_u$**：自适应地为 $\mathbf{p}_{t-1}$ 和 $\mathbf{q}_t$ 分配权重，表示它们对当前任务的重要性。

3. **计算权重$w_u$**：

$$
   w_u = \frac{\exp(\hat{w}_u)}{\sum_u \exp(\hat{w}_u)},\quad \hat{w}_u = \frac{\langle h_2(u), h_3(u) \rangle}{\sqrt{k}}
$$

   - **注意力机制**：使用类似于自注意力的机制，根据输入 $u$ 计算查询（Query）、键（Key）和值（Value）。
   - **前馈网络$h_i(\cdot)$**：简单的单层 MLP，将输入 $u$ 投影到不同的表示空间。
     - $h_1(u)$：生成值（Value）。
     - $h_2(u)$：生成查询（Query）。
     - $h_3(u)$：生成键（Key）。
   - **点积相似度**：计算查询和键之间的相似度 $\hat{w}_u$ ，然后通过 softmax 归一化得到权重 $w_u$ 。

4. **融合输出**：最终的表示 $\mathbf{z}_t$ 是对传递信息和原始信息的加权求和，能够捕获前一个任务对当前任务的影响。

#### 特殊情况

- **第一个任务**：由于没有前一个任务，直接将塔的输出作为 AIT 模块的输出：

$$
  \mathbf{z}_1 = \mathbf{q}_1
$$

#### 预测

- **输出层**：使用一个 MLP 将 $\mathbf{z}_t$ 投影到标量，然后通过 sigmoid 激活函数得到预测概率：

$$
  \hat{y}_t = \text{sigmoid}(\text{MLP}(\mathbf{z}_t))
$$

---

## 3. 行为期望校准器与多任务学习的联合优化

### 交叉熵损失

- **定义**：对于每个任务 $t$，计算预测值 $\hat{y}_t$ 与真实标签 $y_t$ 之间的交叉熵损失：

$$
  \mathcal{L}_{\text{ce}}(\theta) = -\frac{1}{N} \sum_{t=1}^{T} \sum_{(\mathbf{x}, y_t) \in \mathcal{D}} \left( y_t \log \hat{y}_t + (1 - y_t) \log(1 - \hat{y}_t) \right)
$$

  - $N$ 是样本总数。
  - $\theta$ 是模型的参数集合。

### 行为期望校准器

- **动机**：由于序列依赖关系，期望模型的预测也满足 $\hat{y}_{t-1} \geq \hat{y}_t$。然而，模型可能会违反这一约束，因此需要一个机制来校准预测结果。

- **损失函数**：

$$
  \mathcal{L}_{\text{lc}}(\theta) = \frac{1}{N} \sum_{t=2}^{T} \sum_{\mathbf{x} \in \mathcal{D}} \max(\hat{y}_t - \hat{y}_{t-1}, 0)
$$

  - **含义**：当 $\hat{y}_t > \hat{y}_{t-1}$ 时，损失为正，模型受到惩罚；否则，损失为 0。
  - **作用**：通过最小化该损失，模型被鼓励生成满足序列依赖关系的预测结果。

### **总损失函数**

- **组合**：将交叉熵损失和行为期望校准器的损失结合，形成最终的损失函数：

$$
  \mathcal{L}(\theta) = \mathcal{L}_{\text{ce}}(\theta) + \alpha \mathcal{L}_{\text{lc}}(\theta)
$$

  - $\alpha$ 是超参数，控制行为期望校准器的影响力度。
  - **联合优化**：通过同时最小化两个损失，模型既能准确预测各个任务的结果，又能满足序列依赖的约束。

---

## 总结

**AITM 模型**通过以下方式实现了对多步骤转化预测问题的有效建模：

1. **共享嵌入层**：在所有任务之间共享特征表示，促进信息共享，缓解数据不平衡问题。
2. **任务塔结构**：为每个任务建立独立的塔，捕获任务特定的特征，同时保持模型的灵活性和通用性。
3. **自适应信息传递模块（AIT）**：通过注意力机制，在任务之间自适应地传递信息，建模序列依赖关系。
4. **行为期望校准器**：引入额外的损失项，确保模型的预测结果满足实际的序列依赖约束。

---

## 进一步思考

- **超参数$\alpha$的选择**：需要通过实验调整 $\alpha$ 的值，平衡交叉熵损失和校准器损失的影响。
- **塔结构的设计**：虽然本文没有重点讨论塔结构的设计，但选择适合的塔结构（如深度、激活函数、正则化等）仍然对模型性能有重要影响。
- **注意力机制的变体**：可以尝试更复杂的注意力机制（如多头注意力）来增强信息传递的效果。
- **实际应用中的挑战**：在真实的广告和推荐系统中，数据规模大、噪声多，需要考虑模型的可扩展性和鲁棒性。

---
## 代码实现（未考虑缺失标签处理）

*多任务学习中的缺失标签处理*
- **损失函数中加入条件判断**： 在每次计算梯度时，检查每个任务的标签是否缺失。如果某个任务的标签缺失，则跳过该任务的损失计算。
- **动态调整每个任务权重:** 如果某些任务可能会有缺失标签，可以在训练过程中调整每个任务的权重。例如，如果某个任务的标签缺失，你可以将其权重设置为 0，确保该任务的损失不对总损失产生贡献。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Embedding Layer - 将输入特征映射到低维密集向量空间
class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)

    def forward(self, x):
        return self.embedding(x)

# 2. Tower Layer - 用于每个任务的塔结构
class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Tower, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
  
# 3. AIT Module - 自适应信息传递模块
# 注意这里与Transformer中的Self-Attention不同的是，我只需要q_t作为query的最终输出即可
# 所以这里使用的就是最一般的Attention
# Q: [B, n, d] -> [B, 1, d]
# K: [B, m, d] -> [B, 2, d]
# V: [B, m, v] -> [B, 2, d]
class AITModule(nn.Module):
    def __init__(self, input_dim):
        super(AITModule, self).__init__()
        self.h1 = nn.Linear(input_dim, input_dim)
        self.h2 = nn.Linear(input_dim, input_dim)
        self.h3 = nn.Linear(input_dim, input_dim)
        self.w_c = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, p_t_minus_1, q_t):
        """
            p_t_minus_1: [B, dim]
            q_t: [B, dim]
        """
        p_t_minus_1 = p_t_minus_1.unsqueeze(1)  # [B, 1, dim]
        q_t = q_t.unsqueeze(1)  # [B, 1, dim]
        inputs = torch.cat([p_t_minus_1, q_t], dim=1)  # [B, 2, dim]
        
        q = self.h1(q_t)  # [B, 1, d]
        k = self.h2(inputs)  # [B, 2, d]
        v = self.h3(inputs)  # [B, 2, d]

        score = q @ k.permute(0, 2, 1) / q.size(-1) ** 0.5  # [B, 1, 2]
        weight = self.softmax(score)  # [B, 1, 2]

        weighted_v = weight @ v  # [B, 1, dim]
        output = self.w_c(weighted_v.squeeze(1))  # [B, dim]
        return output

# 4. 损失函数类
class AITMLoss(nn.Module):
    def __init__(self, alpha=0.1, num_tasks=4):
        super(AITMLoss, self).__init__()
        self.alpha = alpha
        self.num_tasks = num_tasks

    def forward(self, y_preds, y_true):
        """
            y_preds: [T, B, 1]
            y_true: [B, T]
        """
        # 计算交叉熵损失
        ce_loss = 0
        for i, y_pred in enumerate(y_preds):
            y_true_i = y_true[:, i].unsqueeze(1).float()  # [B, 1]
            ce_loss += F.binary_cross_entropy(y_pred, y_true_i)
        ce_loss /= self.num_tasks  # 对任务数求平均

        # 行为期望校准器损失
        lc_loss = 0
        for i in range(1, len(y_preds)):
            lc_loss += torch.mean(F.relu(y_preds[i] - y_preds[i - 1]))
        
        lc_loss *= self.alpha  # 行为期望校准器的权重

        # 返回总损失
        return ce_loss + lc_loss

# 5. AITM Model - 完整模型
class AITM(nn.Module):
    def __init__(self, input_dim, embed_dim, tower_hidden_dims, num_tasks, alpha):
        super(AITM, self).__init__()
        self.num_tasks = num_tasks 

        # 嵌入层
        self.embedding_layer = EmbeddingLayer(input_dim, embed_dim) 
        # 每个任务的塔结构
        self.towers = nn.ModuleList([Tower(embed_dim, tower_hidden_dims) for _ in range(num_tasks)]) 
        # AIT模块
        self.ait_modules = nn.ModuleList([AITModule(tower_hidden_dims[-1]) for _ in range(1, num_tasks)])
        # 转换层
        self.g_layers = nn.ModuleList([nn.Linear(tower_hidden_dims[-1], tower_hidden_dims[-1]) for _ in range(num_tasks - 1)])
        # 输出层
        self.output_layers = nn.ModuleList([nn.Linear(tower_hidden_dims[-1], 1) for _ in range(num_tasks)])
        # 损失函数
        self.loss_fn = AITMLoss(alpha, num_tasks)

    def forward(self, x):
        embedded = self.embedding_layer(x)
        y_preds = []
        
        # 处理第一个任务
        q_t = self.towers[0](embedded)
        z_t = q_t  # 对于第一个任务，z_t = q_t
        y_t = torch.sigmoid(self.output_layers[0](z_t))
        y_preds.append(y_t)
        
        for i in range(1, self.num_tasks):
            # 获取当前任务的 q_t
            q_t = self.towers[i](embedded)
            # 计算 p_t_minus_1
            p_t_minus_1 = self.g_layers[i - 1](z_t)
            # 应用 AIT 模块
            z_t = self.ait_modules[i - 1](p_t_minus_1, q_t)
            # 输出层
            y_t = torch.sigmoid(self.output_layers[i](z_t))
            y_preds.append(y_t)
        
        return y_preds

    def compute_loss(self, y_preds, y_true):
        return self.loss_fn(y_preds, y_true)

# 7. 模型训练示例
def train_model(model, data_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y_true in data_loader:
            optimizer.zero_grad()
            y_preds = model(x)
            loss = model.compute_loss(y_preds, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader)}")

# 用法示例
input_dim = 1000  # 输入特征维度
embed_dim = 32    # 嵌入维度
tower_hidden_dims = [64, 32]  # 每个塔结构的隐藏层维度列表
num_tasks = 4
alpha = 0.1       # 行为期望校准器的权重
learning_rate = 1e-3

model = AITM(input_dim, embed_dim, tower_hidden_dims, num_tasks, alpha)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```