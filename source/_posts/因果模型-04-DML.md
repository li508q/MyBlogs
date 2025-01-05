---
title: 因果模型-04-DML
date: 2024-11-20 12:00:00
updated: 2024-11-20 12:00:00
tags: 因果模型
categories: 算法
---

## 实际问题

### 问题设定
$$
Y = \theta(X) T + g(X) + \epsilon \quad \text{where} \quad E(\epsilon | T, X) = 0 $$
$$
T = f(X) + \eta \quad \text{where} \quad E(\eta | X) = 0 
$$
如果使用 $X$ 和 $T$ 一起对 $Y$ 建模会存在估计量有偏问题，非渐进正态：
$$
\sqrt{n} (\hat{\theta} - \theta_0) = \left( \frac{1}{n} \sum T_i^2 \right)^{-1} \frac{1}{\sqrt{n}} \sum T_i U_i + \left( \frac{1}{n} \sum T_i^2 \right)^{-1} \left( \frac{1}{\sqrt{n}} \sum T_i (g(x_i) - g(x)) \right)
$$
### 偏差来源

- 部分来自于 $g(X)$ 估计的偏差：残差建模构建内曼正交
- 部分来自于对样本的过拟合：Cross-Fitting

## DML策略

### 1. 结果模型和处理模型得到残差

1. 结果模型
$$
\tilde{Y} = Y - l(x) \quad \text{where} \quad l(x) = E(Y|x)
$$
2. 处理模型
$$
\tilde{T} = T - m(x) \quad \text{where} \quad m(x) = E(T|x)
$$
### 2. 拟合残差

$$\tilde{Y} = \theta(x) \tilde{T} + \epsilon$$
$$\arg\min E[(\tilde{Y} - \theta(x) \cdot \tilde{T})^2]$$
$$E[(\tilde{Y} - \theta(x) \cdot \tilde{T})^2] = E\left(\tilde{T}^2 \left( \frac{\tilde{Y}}{\tilde{T}} - \theta(x) \right)^2\right)$$
$\theta(X)$ 的拟合可以是参数模型也可以是非参数模型
- 参数模型可以直接拟合（*式 1*）
- 非参数模型因为只接受输入和输出，模型 label 变为 $\tilde{Y}/\tilde{T}$，样本权重为 $T^2$（*式2, 3*）
	- 注意这时候所认为的 $\tilde{Y}/\tilde{T}$ 是真实值，预测模型为 $\mu(\tilde{Y}/\tilde{T}|x)$ 

### 3. Cross-Fitting

解决 **Overfitting** 问题，反映在统计学上是解决**收敛速度**的问题。

以 $K=2$ 为例： 
$$I_1, I_2 = sample\_split$$
$$\hat{\theta} = \frac{1}{2} ( \hat{\theta}^{(1)} + \hat{\theta}^{(2)})$$

- **划分数据集**：将数据集分为两个不相交的子集 $I_1$ 和 $I_2$。 

- **第一轮**： 
  - **在 $I_2$ 上估计烦恼参数**：得到 $\hat{l}^{(1)}(X)$ 和 $\hat{m}^{(1)}(X)$。 
  - **在 $I_1$ 上计算残差并估计 $\theta^{(1)}$**。 

- **第二轮**： 
  - **在 $I_1$ 上估计烦恼参数**：得到 $\hat{l}^{(2)}(X)$ 和 $\hat{m}^{(2)}(X)$。 
  - **在 $I_2$ 上计算残差并估计 $\theta^{(2)}$**。 

- **合并结果**： 
$$\hat{\theta} = \frac{1}{2} ( \hat{\theta}^{(1)} + \hat{\theta}^{(2)})$$

