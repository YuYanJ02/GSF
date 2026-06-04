# MCMC EnGMF 稀疏数据轨道确定学习笔记

> **论文**：*MCMC EnGMF for Sparse Data Orbit Determination*  
> **会议/预印本**：AAS SciTech 2023，论文编号 AAS 23-356（Preprint）  
> **作者**：Dalton Durant, Andrey A. Popov, Renato Zanetti（The University of Texas at Austin）  
> **原文 PDF**：[`MCMC ENGMF FOR SPARSE DATA ORBIT DETERMINATION.pdf`](MCMC%20ENGMF%20FOR%20SPARSE%20DATA%20ORBIT%20DETERMINATION.pdf)  
> **整理依据**：[`MCMC-EnGMF.txt`](MCMC-EnGMF.txt)（EnGMF / Batch EnGMF / MCMC EnGMF 中文详解）  
>
> **阅读说明**：GitHub 官方行内公式语法为 `` $`...`$ ``（反引号包裹，避免 `_` 被 Markdown 吃掉）；块级公式用独立 `` ```math `` 代码块。

---

## EnGMF 讲解：动机、直觉与公式推导

EnGMF（Ensemble Gaussian Mixture Filter，集成高斯混合滤波器）是一种用于非线性、非高斯系统的序贯蒙特卡洛滤波方法。本节围绕论文第3-5页的内容，从动机直觉到数学公式推导逐步展开。

### 一、为什么需要 EnGMF？

在轨道确定问题中，系统是非线性的（二体运动、角度测量等），而且由于稀疏观测和观测盲区，状态的概率密度函数（PDF）常常呈现**非高斯**形态。传统滤波器如扩展卡尔曼滤波（EKF）或无迹卡尔曼滤波（UKF）假设状态为高斯分布，在强非线性下会失效。粒子滤波虽然能表示任意分布，但存在粒子退化和计算负担大的问题。

EnGMF 的核心思想是：**用一簇高斯分布的加权和（高斯混合模型，GMM）来近似状态的后验分布**。每个高斯分量可以看作一个“假设”，整体既能捕捉多模态或非高斯特征，又比纯粒子滤波更平滑、更稳定。

### 二、EnGMF 的数学框架

系统模型（论文公式 (1)）：

```math
\begin{aligned}
\mathbf{x}_k &= f(\mathbf{x}_{k-1}) + \eta_k, \\
\mathbf{y}_k &= h(\mathbf{x}_k) + \epsilon_k,
\end{aligned}
```

其中 $`\eta_k \sim \mathcal{N}(0,Q)`$， $`\epsilon_k \sim \mathcal{N}(0,R)`$。论文中设 $`Q = 0`$（无过程噪声），以突出稀疏观测下的估计能力。

#### 1. 先验经验分布（公式 (2)）

在 $`k-1`$ 时刻，我们用 $`N`$ 个独立同分布的粒子 $`\{\mathbf{x}_{k-1}^{(i)}\}`$ 近似真实分布：

```math
p(\mathbf{x}_{k-1}) \approx \sum_{i=1}^{N}\frac{1}{N}\delta(\mathbf{x}_{k-1} - \mathbf{x}_{k-1}^{(i)}),
```

$`\delta(\cdot)`$ 是狄拉克函数。实际计算中，我们把它视作协方差趋于零的高斯分布：

```math
p(\mathbf{x}_{k-1}) \approx \lim_{\mathbf{P}_{k-1}\to 0}\sum_{i=1}^{N}\frac{1}{N}\mathcal{N}(\mathbf{x}_{k-1}; \mathbf{x}_{k-1}^{(i)}, \mathbf{P}_{k-1}).
```

**直觉**：粒子集合代表离散的“点质量”，但为了后续用高斯混合处理，需要给每个点赋予一个微小的高斯核。

#### 2. 预测步骤：粒子传播与 KDE 构造 GMM（公式 (3)）

将每个粒子按动力学 $`f`$ 传播到下一时刻 $`k`$，得到 $`\bar{\mathbf{x}}_k^{(i)} = f(\mathbf{x}_{k-1}^{(i)})`$。之后，使用核密度估计（KDE）将粒子集转化为连续的高斯混合模型：

```math
p(\mathbf{x}_k) \approx \sum_{i=1}^{N}\frac{1}{N}\mathcal{N}(\mathbf{x}_k; \bar{\mathbf{x}}_k^{(i)}, B).
```

带宽矩阵 $`B = \beta \bar{\mathbf{P}}_k`$，其中 $`\bar{\mathbf{P}}_k`$ 是传播后粒子的样本协方差， $`\beta > 0`$ 是带宽参数（通常由 Silverman 法则确定）。

**直觉**：如果只用粒子点，似然计算困难且易退化。给每个粒子周围加一个高斯核，就把离散点变成了一个连续、平滑的分布。带宽 $`B`$ 控制平滑程度：太大则分布过平（丢失细节），太小则粒子孤岛化。KDE 相当于“把每个粒子当作一个高斯假设”，从而自然获得 GMM 形式。

#### 3. 更新步骤：利用测量加权（公式 (4)-(7)）

有了先验 GMM 后，获得新的测量 $`\mathbf{y}_k`$，计算后验 GMM：

```math
p(\mathbf{x}_k \mid \mathbf{y}_k) \approx \sum_{i=1}^{N} w_k^{(i)} \mathcal{N}(\mathbf{x}_k; \hat{\mathbf{x}}_k^{(i)}, \hat{\mathbf{P}}_k^{(i)}).
```

每个粒子的高斯分量独立进行 EKF 更新（线性化测量模型），得到后验均值和协方差 $`\hat{\mathbf{x}}_k^{(i)}, \hat{\mathbf{P}}_k^{(i)}`$。权重 $`w_k^{(i)}`$ 反映了该粒子的测量似然：

```math
w_k^{(i)} = \frac{w_{k-1}^{(i)} \; \mathcal{N}\bigl(\mathbf{y}_k; h(\hat{\mathbf{x}}_k^{(i)}), H_k^{(i)} B (H_k^{(i)})^T + R\bigr)}{\sum_{j=1}^{N} w_{k-1}^{(j)} \; \mathcal{N}\bigl(\mathbf{y}_k; h(\hat{\mathbf{x}}_k^{(j)}), H_k^{(j)} B (H_k^{(j)})^T + R\bigr)}.
```

其中 $`H_k^{(i)}`$ 是 $`h`$ 在 $`\hat{\mathbf{x}}_k^{(i)}`$ 处的雅可比矩阵。

**直觉**：EKF 更新每个高斯分量，得到后验高斯。然后根据预测的测量分布（高斯）与实际测量 $`\mathbf{y}_k`$ 的匹配程度，重新分配权重。匹配越好，权重越大。这里的协方差中多了一项 $`H B H^T`$，它来自先验 GMM 的核宽度带来的额外不确定性。

#### 4. 重采样

为了得到下一时刻的粒子集合，从后验 GMM 中重新采样 $`N`$ 个粒子（方法见文献[18]）。这些粒子再用于下一轮迭代。

整体状态估计为加权平均：

```math
\hat{\mathbf{x}}_k = \sum_{i=1}^{N} w_k^{(i)} \hat{\mathbf{x}}_k^{(i)},\qquad
\hat{\mathbf{P}}_k = \sum_{i=1}^{N} w_k^{(i)} \bigl[ \hat{\mathbf{P}}_k^{(i)} + \hat{\mathbf{x}}_k^{(i)}(\hat{\mathbf{x}}_k^{(i)})^T - \hat{\mathbf{x}}_k \hat{\mathbf{x}}_k^T \bigr].
```

### 三、算法步骤总结

1. **预测**：将上一时刻粒子 $`x_{k-1}^{(i)}`$ 通过动力学传播得到 $`x_k^{(i)}`$。
2. **KDE 化**：用核密度估计将粒子集转化为 GMM，带宽 $`B = \beta \bar{P}_k`$。
3. **更新**：对每个高斯分量进行 EKF 更新，得到后验均值和协方差；根据测量似然计算权重。
4. **重采样**：从后验 GMM 中抽取 $`N`$ 个粒子作为下一时刻的先验粒子。

### 四、关键参数：带宽 $`B`$ 与 Silverman 法则

带宽矩阵 $`B = \beta \bar{\mathbf{P}}_k`$ 控制每个高斯核的展宽。论文采用 **Silverman 经验法则**： $`\beta = \left( \frac{4}{n_x+2} \right)^{\frac{2}{n_x+4}} N^{-\frac{2}{n_x+4}}`$（$`n_x`$ 为状态维数）。该公式假设数据来自高斯分布，能给出最小均方误差下的最优带宽。如果实际分布偏离高斯，带宽可能不最优，但 EnGMF 仍保持鲁棒。

**直觉**：带宽太大会使分布过于平滑，模糊真实非高斯特征；太小则每个粒子独立成峰，退化为粒子滤波。Silverman 法则提供了一个自动、合理的初始值。

### 五、与粒子滤波和高斯和滤波的对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| 粒子滤波 | 可表示任意分布 | 粒子退化、需要大量粒子 |
| 高斯和滤波 | 连续、可解析更新 | 需要事先指定混合分量数 |
| **EnGMF** | 结合两者：用 KDE 从粒子生成 GMM，兼具灵活性与平滑性 | 需要调整带宽，计算略重于标准粒子滤波 |

EnGMF 本质上是一种 **带有自适应核密度的粒子滤波** —— 它用高斯核“模糊”粒子，从而抑制退化，同时保留对非高斯多模态分布的表示能力。

---

## Batch EnGMF 讲解：动机、直觉与公式推导

Batch EnGMF 是在标准 EnGMF 基础上提出的一种改进方案，旨在解决**稀疏观测轨道确定中 EnGMF 的保守性以及计算效率问题**。

### 一、动机：为什么要把一整段测量“批处理”成一个高斯？

标准 EnGMF 虽然能处理非高斯分布，但它在处理**单个原始测量**时存在两个问题：

1. **带宽选择的次优性**：EnGMF 在每个测量时刻都需要用 Silverman 法则计算带宽 $`B = \beta \bar{\mathbf{P}}_k`$。该法则隐含假设粒子分布是**高斯**的。然而，在稀疏观测下，经过非线性动态传播后的先验分布往往是非高斯的（例如“香蕉形”或“多模态”）。此时 Silverman 法则给出的带宽偏大，导致滤波结果**保守**（估计协方差偏大，SNEES < 1）。

2. **计算负担大**：每次原始测量都要进行 KDE 构造、EKF 更新和重采样，而一个 track（测量批次）可能包含几十个点（如 2 分钟每隔 10 秒共 12 个测量），逐序处理的累积计算量可观。

**Batch EnGMF 的核心想法**：  
将一个 track 内所有的原始测量（如 $`\rho,\dot{\rho},\alpha,\delta`$ 或角度只测）**一次性**通过**批处理最小二乘信息滤波器**，压缩成一个**状态空间中的高斯分布**（均值 $`\hat{\mathbf{x}}^{\text{batch}}`$，协方差 $`\hat{\mathbf{P}}^{\text{batch}}`$）。然后将这个高斯分布作为**虚拟测量**输入到 EnGMF 中。好处：

- 虚拟测量是**高斯**的（因批处理假设线性化后误差为高斯），更符合 Silverman 法则的假设，减少保守性。
- 一个 track 只做一次批处理，然后 EnGMF 只做一次更新（而不是 $`K`$ 次），计算量大幅下降（论文图 5(c) 显示 Batch EnGMF 比 EnGMF 快约 2 倍）。
- 批处理利用了整个 track 的几何信息，能更准确地估计状态，尤其当测量可观测（如全测量向量）时精度更高。

### 二、Batch EnGMF 的算法流程

分为四个主要步骤：

#### 步骤 1：粒子传播至 track 起始时刻

当前有 $`N`$ 个粒子 $`\{\mathbf{x}_{p-1}^{(i)}\}`$（上一 track 结束后的后验粒子）。将它们分别用动力学 $`f`$ 传播到下一测量 track $`p`$ 的第一个测量时刻 $`t_{p,1}`$，得到 $`\bar{\mathbf{x}}_p^{(i)}`$。

#### 步骤 2：构建先验 GMM 并计算样本均值（作为批处理的先验）

使用 KDE 将传播后的粒子转化为 GMM：

```math
p(\mathbf{x}_p) \approx \sum_{i=1}^{N}\frac{1}{N}\mathcal{N}(\mathbf{x}_p; \bar{\mathbf{x}}_p^{(i)}, B),\quad B = \beta \bar{\mathbf{P}}_p.
```

然后计算样本均值 $`\bar{\mathbf{x}}_{p,0}^{\text{batch}}`$ 作为批处理的状态先验：

```math
\bar{\mathbf{x}}_{p,0}^{\text{batch}} = \frac{1}{N}\sum_{i=1}^{N} \bar{\mathbf{x}}_p^{(i)}.
```

**关键点**：论文中为批处理设置的**先验协方差逆矩阵为 0**，即 $`(\bar{\mathbf{P}}_{p,0}^{\text{batch}})^{-1} = O`$（公式 (12)）。这表示一个**无信息（扩散）先验**——批处理完全依赖测量信息，不依赖先验协方差，避免因先验协方差不准确而引入偏差。

#### 步骤 3：运行批处理最小二乘信息滤波器

论文采用**批次最小二乘信息滤波器**（表 1 伪代码）。其数学基础是**正规方程**（公式 (9)）：

```math
(H^T R^{-1} H + P_0^{-1}) \hat{\mathbf{x}} = H^T R^{-1} \mathbf{y} + P_0^{-1} \bar{\mathbf{x}}_0.
```

在 Batch EnGMF 中：
- $`P_0^{-1} = 0`$（无信息先验）
- $`H`$ 是**所有观测相对于状态的雅可比矩阵**（将每个原始测量映射到状态）
- $`R`$ 是测量噪声协方差矩阵（块对角，包含每个 $`\rho,\dot{\rho},\alpha,\delta`$ 的方差）

因此正规方程简化为 $`H^T R^{-1} H \hat{\mathbf{x}} = H^T R^{-1} \mathbf{y}`$，即经典**加权最小二乘**解。  
表 1 中的算法还处理了动力学映射（通过状态转移矩阵 $`\Phi`$），将每个测量时刻的残差映射回 track 起始时刻。最终输出一个高斯估计： $`\hat{\mathbf{x}}_{p,1}^{\text{batch}}`$ 和 $`\hat{\mathbf{P}}_{p,1}^{\text{batch}}`$。

#### 步骤 4：将批处理结果作为虚拟测量更新 EnGMF

将 $`\hat{\mathbf{x}}_{p,1}^{\text{batch}}`$ 视为一个**虚拟测量**，测量模型为 $`\mathbf{y} = \mathbf{x} + \epsilon`$， $`\epsilon \sim \mathcal{N}(0,\hat{\mathbf{P}}_{p,1}^{\text{batch}})`$，即 $`h(\mathbf{x}) = \mathbf{x}`$， $`H = I`$， $`R = \hat{\mathbf{P}}_{p,1}^{\text{batch}}`$。

对**每一个粒子** $`i`$ 执行卡尔曼更新（因为 $`h`$ 线性）：

```math
\begin{aligned}
\mathbf{K} &= B (B + \hat{\mathbf{P}}_{p,1}^{\text{batch}})^{-1},\\
\hat{\mathbf{x}}_p^{(i)} &= \bar{\mathbf{x}}_p^{(i)} + \mathbf{K}\bigl( \hat{\mathbf{x}}_{p,1}^{\text{batch}} - \bar{\mathbf{x}}_p^{(i)} \bigr),\\
\hat{\mathbf{P}}_p^{(i)} &= (I - \mathbf{K}) B.
\end{aligned}
```

权重更新公式（论文 (16)）：

```math
w_p^{(i)} = \frac{w_{p-1}^{(i)} \; \mathcal{N}\bigl( \hat{\mathbf{x}}_{p,1}^{\text{batch}};\, \bar{\mathbf{x}}_p^{(i)},\, B + \hat{\mathbf{P}}_{p,1}^{\text{batch}} \bigr)}{\sum_{j=1}^{N} w_{p-1}^{(j)} \; \mathcal{N}\bigl( \hat{\mathbf{x}}_{p,1}^{\text{batch}};\, \bar{\mathbf{x}}_p^{(j)},\, B + \hat{\mathbf{P}}_{p,1}^{\text{batch}} \bigr)}.
```

最后，从后验 GMM 中重采样 $`N`$ 个粒子，进入下一 track 的传播。

### 三、核心公式推导与直觉

1. **为什么批处理能产生“更高斯”的分布？**  
   原始测量在时间序列上可能存在不可观测性（如只有角度），逐序滤波时中间状态分布会变得严重非高斯。批处理一次性利用全部测量，通过非线性最小二乘迭代，将测量信息映射回 track 起始时刻的状态。在线性化误差较小的条件下（全测量向量时），该状态的估计误差近似服从高斯分布（中心极限定理）。因此，虚拟测量及其协方差形成一个合理的高斯近似。

2. **为什么采用无信息先验（$`P_0^{-1}=0`$）？**  
   如果给批处理一个非零先验协方差，可能会与 EnGMF 的当前粒子集合所隐含的分布发生冲突。粒子集合已经代表了轨道状态的不确定性，批处理阶段应该只利用当前 track 的测量信息，不掺杂历史信息。因此设 $`P_0^{-1}=0`$，使批处理完全由测量驱动。

3. **带宽 $`B`$ 在更新中的作用**  
   在权重公式 $`\mathcal{N}(\hat{\mathbf{x}}^{\text{batch}}; \bar{\mathbf{x}}_p^{(i)}, B + \hat{\mathbf{P}}^{\text{batch}})`$ 中， $`B`$ 反映了先验 GMM 的核宽度。如果不加 $`B`$，会导致过分信任先验粒子的位置，低估不确定性。增加 $`B`$ 相当于在先验粒子周围涂抹了一层“不确定性糊”，使得离群粒子也有机会得到较大权重。

### 四、总结直觉

> **Batch EnGMF = 把一段原始测量“浓缩”成一个高斯球，然后让 EnGMF 一次性“吸收”这个球。**

- **为什么浓缩？** 原始测量太多、太快，逐序处理既慢又容易产生不良的 KDE 带宽。
- **为什么能浓缩？** 在一段可观测的轨道弧段内，批量最小二乘估计的误差近似为高斯。
- **代价是什么？** 当观测信息不足（角度只测、弧段太短或间隔太长）时，单高斯近似失效。

Batch EnGMF 在**传感器提供全测量向量（$`\rho,\dot{\rho},\alpha,\delta`$）** 且 **跟踪相对频繁** 的场景下表现优异；对于纯角度跟踪或极其稀疏的数据，需要 MCMC EnGMF 的 GMM 表达能力。

---

## MCMC EnGMF 讲解：动机、直觉与公式推导

MCMC EnGMF（Markov Chain Monte Carlo Ensemble Gaussian Mixture Filter）是论文的核心贡献，旨在解决 **Batch EnGMF 在处理不可观测系统（如纯角度测量）时单高斯近似失效** 的问题。当测量不能提供足够信息（只有角度，无距离和距离变化率）且轨道间隔较大时，由一段跟踪弧段反演出的状态分布会呈现**强非高斯性**（例如香蕉形、弯曲流形甚至多峰）。此时，单个高斯分布无法描述这种不确定性，导致 Batch EnGMF 估计迅速发散。而 MCMC EnGMF 通过 MCMC 采样，将整段测量转换成一个**高斯混合模型（GMM）** 作为虚拟测量，从而保留了分布的非高斯特征。

### 一、为什么需要 MCMC？

#### 1.1 角度跟踪的不可观测性问题

考虑纯角度测量（右赤经 $`\alpha`$ 和赤纬 $`\delta`$）：
- 一个地面站仅提供视线方向，不提供距离信息。
- 一条短弧（如 2 分钟，12 个测量点）不足以唯一确定轨道——存在多族可能的轨道（距离和速度耦合）。
- 通过动力学传播到弧段起点，可能的状态分布不是一个椭圆（高斯），而是一条**弯曲的狭长流形**或**分离的模态**。

#### 1.2 单高斯近似的失败

Batch EnGMF 试图用一个高斯 $`\mathcal{N}(\hat{\mathbf{x}}^{\text{batch}},\hat{\mathbf{P}}^{\text{batch}})`$ 来代表整个弧段的“压缩测量”。对于角度跟踪，这种近似会严重高估某些方向的不确定性（将弯曲分布拉直成椭圆），或者低估其他方向的不确定性（覆盖不到真实模态）。结果如论文图 7(a) 所示：当轨道间隔超过 4 个周期后，Batch EnGMF 的 RMSE 急剧发散。

#### 1.3 MCMC 的解决方案

MCMC EnGMF 的核心思想：**不把一个 track 压缩成一个高斯，而是压缩成一个高斯混合模型（GMM）**。具体地：
1. 使用 MCMC（Metropolis-Hastings 算法）从该 track 的后验分布 $`p(\mathbf{x} \mid \mathbf{Y}_p)`$ 中抽取 $`M`$ 个样本。
2. 将这些样本作为 GMM 的中心，用核密度估计（KDE）构成连续的 GMM。
3. 将这个 GMM 作为虚拟测量输入 EnGMF。每个虚拟测量分量 $`\mathbf{x}_m^{\text{MCMC}}`$ 都有自己的概率权重（$`1/M`$）。

这样，EnGMF 就能同时处理多个“假设”状态，捕捉非高斯结构。

### 二、MCMC EnGMF 的整体流程

与 Batch EnGMF 类似，但用 MCMC 生成 GMM 代替单个高斯。

#### 步骤 1：粒子传播至 track 起始时刻

将上一 track 结束后的 $`N`$ 个粒子 $`\{\mathbf{x}_{p-1}^{(i)}\}`$ 传播到当前 track $`p`$ 的第一个测量时刻，得到 $`\bar{\mathbf{x}}_p^{(i)}`$。

#### 步骤 2：对 track 进行 MCMC 采样，生成 GMM

##### 2.1 构建 M-H 的提议分布（公式 (18)）

论文使用 **EnKF（集合卡尔曼滤波）** 为 M-H 算法提供一个高斯提议分布。对传播后的粒子集合 $`\{\bar{\mathbf{x}}_p^{(i)}\}`$ 运行一次 EnKF（以第一个测量时刻的角度测量为输入），得到状态估计 $`\hat{\mathbf{x}}_{p,1}^{\text{EnKF}}`$ 和协方差 $`\mathbf{P}_{p,1}^{\text{EnKF}}`$。然后定义对称提议分布（高斯）：

```math
q(\mathbf{x}' \mid \mathbf{x}) = \mathcal{N}(\mathbf{x}'; \mathbf{x}, \mathbf{P}_{p,1}^{\text{EnKF}}),\quad
q(\mathbf{x} \mid \mathbf{x}') = \mathcal{N}(\mathbf{x}; \mathbf{x}', \mathbf{P}_{p,1}^{\text{EnKF}}).
```

**直觉**：EnKF 利用全部测量提供了一个合理的初始猜测（均值）和尺度（协方差）。对称提议简化了接受率计算（$`q(\mathbf{x}' \mid \mathbf{x}) = q(\mathbf{x} \mid \mathbf{x}')`$）。

##### 2.2 用 Metropolis-Hastings 从后验分布采样

目标分布是给定整段测量 $`\mathbf{Y}_p = \{\mathbf{y}_{p,k}\}_{k=1}^{K}`$ 的后验分布（假设无信息先验）：

```math
p(\mathbf{x} \mid \mathbf{Y}_p) \propto p(\mathbf{Y}_p \mid \mathbf{x}) = \prod_{k=1}^{K} \exp\!\left( -\frac{1}{2} \bigl[ \mathbf{y}_{p,k} - h(f(\mathbf{x})_k) \bigr]^T \mathbf{R}_p^{-1} \bigl[ \mathbf{y}_{p,k} - h(f(\mathbf{x})_k) \bigr] \right).
```

M-H 算法：
1. 初始化 $`x = x_0`$（取 EnKF 估计值）。
2. 从提议分布 $`q(x' \mid x)`$ 抽一个新样本 $`x'`$。
3. 计算接受率：

```math
\alpha = \min\left(1,\; \frac{p(x' \mid \mathbf{Y}_p)\, q(x \mid x')}{p(x \mid \mathbf{Y}_p)\, q(x' \mid x)}\right).
```

   由于提议对称， $`q(x \mid x') = q(x' \mid x)`$，因此 $`\alpha = \min\left(1,\; \frac{p(x' \mid \mathbf{Y}_p)}{p(x \mid \mathbf{Y}_p)}\right)`$。
4. 以概率 $`\alpha`$ 接受 $`x'`$，否则保持 $`x`$。
5. 重复直到收集到足够样本（$`L`$ 次接受，论文取 $`L`$ 为链长，最终只保留每个链的最后状态）。

对每个 MCMC 分量 $`m = 1 \dots M`$ 独立运行一条链（或使用多条并行链），得到 $`M`$ 个状态样本 $`\{\mathbf{x}_m^{\text{MCMC}}\}`$。

##### 2.3 从样本构建 GMM（公式 (21)）

使用 KDE 将 $`M`$ 个样本转化为 GMM：

```math
p(\mathbf{x}^{\text{MCMC}}) \approx \sum_{m=1}^{M} \frac{1}{M} \mathcal{N}(\mathbf{x}^{\text{MCMC}}; \mathbf{x}_m^{\text{MCMC}}, B^{\text{MCMC}}),
```

带宽矩阵 $`B^{\text{MCMC}} = \beta^{\text{MCMC}} \hat{\mathbf{P}}^{\text{MCMC}}`$， $`\hat{\mathbf{P}}^{\text{MCMC}}`$ 是样本协方差， $`\beta^{\text{MCMC}}`$ 按 Silverman 法则由 $`M`$ 计算。

**关键区别**：Batch EnGMF 输出一个高斯（$`M=1`$），MCMC EnGMF 输出一个有 $`M`$ 个分量的 GMM。

#### 步骤 3：用 GMM 虚拟测量更新 EnGMF

虚拟测量是一个 **GMM 集合** $`\{\mathbf{x}_m^{\text{MCMC}}\}_{m=1}^{M}`$，每个分量的协方差为 $`B^{\text{MCMC}}`$，测量模型 $`h(\mathbf{x}) = \mathbf{x}`$， $`H = I`$，测量噪声协方差 $`R = B^{\text{MCMC}}`$。

EnGMF 的更新需要对 **每个粒子 $`i`$** 和 **每个 MCMC 分量 $`m`$** 进行联合更新：

先验 GMM（来自传播后的粒子）有 $`N`$ 个分量；虚拟测量 GMM 有 $`M`$ 个分量；因此后验 GMM 规模为 $`N \times M`$：

```math
p(\mathbf{x}_p \mid \mathbf{Y}) \approx \sum_{i=1}^{N} \sum_{m=1}^{M} w_p^{(i,m)} \; \mathcal{N}(\mathbf{x}_p; \hat{\mathbf{x}}_p^{(i,m)}, \hat{\mathbf{P}}_p^{(i,m)}).
```

对每一个 $`(i,m)`$ 对：
- 卡尔曼更新（$`h=I`$）：

```math
\begin{aligned}
\mathbf{K} &= B (B + B^{\text{MCMC}})^{-1},\\
\hat{\mathbf{x}}_p^{(i,m)} &= \bar{\mathbf{x}}_p^{(i)} + \mathbf{K}\bigl( \mathbf{x}_m^{\text{MCMC}} - \bar{\mathbf{x}}_p^{(i)} \bigr),\\
\hat{\mathbf{P}}_p^{(i,m)} &= (I - \mathbf{K}) B.
\end{aligned}
```

- 权重更新（公式 (25)）：

```math
w_p^{(i,m)} = \frac{w_{p-1}^{(i,m)} \; \mathcal{N}\bigl( \mathbf{x}_m^{\text{MCMC}};\, \bar{\mathbf{x}}_p^{(i)},\, B + B^{\text{MCMC}} \bigr)}{\sum_{i=1}^{N}\sum_{m=1}^{M} w_{p-1}^{(i,m)} \; \mathcal{N}\bigl( \mathbf{x}_m^{\text{MCMC}};\, \bar{\mathbf{x}}_p^{(i)},\, B + B^{\text{MCMC}} \bigr)}.
```

**直觉**：每个 MCMC 样本代表了一个可能的“真实轨道”假设，每个粒子代表当前对该假设的置信度。两者的组合会产生一个丰富的后验分布，能够覆盖弯曲流形或多峰结构。

#### 步骤 4：重采样

从 $`N \times M`$ 的后验 GMM 中抽取 $`N`$ 个粒子（不是 $`N \times M`$），作为下一 track 的初始粒子（方法见文献[18]）。

### 三、核心公式的详细推导与动机

1. **为什么用 Metropolis-Hastings 而不是直接计算后验？**  
   对于纯角度跟踪，后验 $`p(\mathbf{x} \mid \mathbf{Y}_p)`$ 没有解析形式，且维度高（6维）、非线性强。M-H 是一种通用 MCMC 方法，只需知道未归一化的后验密度（即似然乘积），就能通过构造马尔可夫链渐近地从真实后验中采样。常数 $`c`$ 在计算接受率时会约掉，因此不需要计算归一化常数。

2. **为什么用 EnKF 作为提议分布？**  
   提议分布 $`q(x' \mid x) = \mathcal{N}(x'; x, P_{\text{EnKF}})`$ 是随机游走的一种：新样本在当前样本上加一个高斯扰动。好处：尺度匹配（$`P_{\text{EnKF}}`$ 给出了后验分布的大致尺度，避免步长不当）、计算效率（EnKF 一次计算即可）、对称性（简化接受率为似然比）。初始状态 $`x_0 = \hat{x}_{\text{EnKF}}`$ 使链从高概率区域开始，缩短 burn-in 时间。

3. **为什么从每条链只取最后一个状态？**  
   表 2 伪代码中，每条链只保存接受 $`L`$ 次后的最后一个状态作为 GMM 的一个样本。这是为了减少自相关性。如果收集整个链，相邻样本高度相关，会导致 GMM 协方差被低估。取最后状态（经过足够多步后认为独立）可以获得近似独立的样本。

4. **带宽 $`B^{\text{MCMC}}`$ 的含义**  
   $`B^{\text{MCMC}}`$ 用于平滑 MCMC 样本生成的 GMM。尽管样本本身来自后验，但数量 $`M`$ 有限，加一个小的核可以避免 GMM 出现空洞，使 EnGMF 更新更平滑。论文使用 Silverman 法则根据 $`M`$ 确定 $`\beta^{\text{MCMC}}`$。

### 四、与 Batch EnGMF 的对比

| 特性 | Batch EnGMF | MCMC EnGMF |
|------|-------------|-------------|
| 虚拟测量形式 | 单个高斯 | GMM（$`M`$ 个高斯） |
| 更新规模 | $`N`$（粒子数） | $`N \times M`$（粒子数×MCMC 样本数） |
| 计算速度（相对 EnGMF） | 快 2 倍 | 慢 2 倍（当 $`M=200`$） |
| 角度跟踪（间隔 $`>4`$ 周期） | 发散 | 稳定、高精度 |
| 适用场景 | 可观测系统（全测量） | 不可观测系统（纯角度） |

论文图 7(a) 显示：当轨道间隔超过 4 个周期，Batch EnGMF 的 RMSE 直线上升，而 MCMC EnGMF 保持较低且缓慢增长；图 7(b) 显示其 SNEES 更接近 1，一致性更好。

### 五、关键参数分析：MCMC 样本数 $`M`$ 的选择

论文第 17–18 页专门分析了 $`M`$ 的影响（图 6）：
- 当 $`M < 100`$，RMSE 随 $`M`$ 增加而显著下降（样本不足，GMM 不能充分描述非高斯性）。
- 当 $`M \geq 100`$，RMSE 基本稳定，但计算时间线性增长。
- 因此论文选择 $`M = 200`$ 作为安全启发值：在精度和速度间平衡。

**直觉**： $`M`$ 决定了 GMM 表示真实后验的精细度。对于角度跟踪，后验可能是一条细长弯曲流形，需要至少数十个样本才能覆盖其主要模式；200 个样本足以捕捉结构，且计算负担可接受（相比 EnGMF 只慢 2 倍）。

### 六、总结直觉

> **MCMC EnGMF = 用 MCMC 采样将一段角度测量“雕刻”成一个非高斯的 GMM 雕像，然后让 EnGMF 的每个粒子都与这个雕像的每个“特征点”进行匹配。**

- **为什么需要雕刻？** 因为角度测量产生的是弯曲流形，而非椭圆。
- **为什么用 MCMC？** 因为无法解析计算该流形的形状，M-H 算法能从中采样。
- **为什么最后还要用 EnGMF？** 因为我们需要将这一测量信息与历史状态信息（粒子集）融合，并且要保持序贯滤波结构。

该方法在论文的数值结果中证明了对于纯角度稀疏轨道确定的优越性，尽管计算开销增加，但换来了鲁棒性和一致性。未来工作可考虑用更高效的 MCMC 变体（如 HMC）加速。

---

## 附录：Metropolis-Hastings 中关于提议分布的补充说明

### 一、对称提议分布详解（针对 MCMC EnGMF 中的 Metropolis-Hastings）

#### 1. 什么是提议分布 $`q(\mathbf{x}' \mid \mathbf{x})`$？

在 M-H 算法中，我们想从一个复杂的**目标分布** $`p(x)`$ 中抽样（这里 $`p(x) = p(x \mid \mathbf{Y}_p)`$，即给定测量后的状态后验）。我们无法直接抽样，所以构造一条马尔可夫链：
- 当前状态为 $`x`$，
- 从一个简单的**提议分布** $`q(x' \mid x)`$ 中抽一个候选状态 $`x'`$，
- 以一定概率接受 $`x'`$ 作为链的下一个状态。

$`q(x' \mid x)`$ 可以任意选择，常见的有：
- **随机游走**： $`x' = x + \text{噪声}`$，例如 $`q(x' \mid x) = \mathcal{N}(x'; x, \Sigma)`$。
- **独立采样**：与当前状态无关，如 $`q(x' \mid x) = \mathcal{N}(x'; \mu, \Sigma)`$。

#### 2. 对称提议分布的定义

**对称提议分布**是指满足

```math
q(x' \mid x) = q(x \mid x')
```

对所有可能的 $`x, x'`$ 成立。换句话说，从 $`x`$ 跳到 $`x'`$ 的概率密度，等于从 $`x'`$ 跳回 $`x`$ 的概率密度。**直观**：提议的“跳跃”没有方向性，来回概率相等。

#### 3. 对称性带来的简化（关键！）

在标准的 Metropolis-Hastings 算法中，接受率公式为：

```math
\alpha = \min\left(1,\; \frac{p(x') \, q(x \mid x')}{p(x) \, q(x' \mid x)}\right).
```

当提议分布对称时， $`q(x \mid x') = q(x' \mid x)`$，两者约掉，接受率简化为：

```math
\alpha = \min\left(1,\; \frac{p(x')}{p(x)}\right).
```

这意味着只需比较目标分布（后验）的概率密度比值，完全不用计算提议分布的值。这极大简化了计算，尤其在每步都要调用昂贵动力学模型（如轨道传播）时非常有价值。

#### 4. 论文中的对称提议分布（公式 (18)）

论文中使用的提议分布是：

```math
q(\mathbf{x}' \mid \mathbf{x}) = \mathcal{N}(\mathbf{x}'; \mathbf{x}, \mathbf{P}_{p,k=1}^{\text{EnKF}}),\quad
q(\mathbf{x} \mid \mathbf{x}') = \mathcal{N}(\mathbf{x}; \mathbf{x}', \mathbf{P}_{p,k=1}^{\text{EnKF}}).
```

这两个分布显然相等，因为高斯分布只依赖于差值的马氏距离：

```math
\mathcal{N}(\mathbf{x}'; \mathbf{x}, \mathbf{P}) = \frac{1}{\sqrt{(2\pi)^d |\mathbf{P}|}} \exp\!\left( -\frac{1}{2}(\mathbf{x}'-\mathbf{x})^T \mathbf{P}^{-1} (\mathbf{x}'-\mathbf{x}) \right),
```

交换 $`\mathbf{x}`$ 和 $`\mathbf{x}'`$ 后指数完全相同。所以确实是对称的。**随机游走 Metropolis** 就是这种形式：候选状态 = 当前状态 + 零均值高斯噪声。它是典型的对称提议。

#### 5. 为什么选择对称提议？

- **计算效率**：不需要每次计算提议分布的值，只需计算目标分布 $`p(\cdot)`$ 的比值。在轨道确定中， $`p(x')`$ 需要把 $`x'`$ 通过动力学传播到每个测量时刻并计算似然，这很耗时；省去提议分布的计算可节省约一半成本。
- **易于实现**：只需要指定一个协方差矩阵 $`\mathbf{P}^{\text{EnKF}}`$，该矩阵由 EnKF 自动估计给出，无需手动调节缩放因子。
- **足够有效**：对于后验分布接近高斯或单峰但略有扭曲的情形，随机游走 Metropolis 混合效率良好。论文中的角度跟踪问题，后验往往是细长弯曲流形，对称高斯提议仍然能较好地探索（只要步长合适）。

### 二、为什么 Metropolis-Hastings 算法需要“提议分布”？

#### 1. 直接抽样为什么不行？

在论文中，目标分布是：

```math
p(\mathbf{x} \mid \mathbf{Y}_p) \propto \prod_{k=1}^{K} \exp\!\left( -\frac{1}{2} \bigl[ \mathbf{y}_{p,k} - h(f(\mathbf{x})_k) \bigr]^T \mathbf{R}_p^{-1} \bigl[ \mathbf{y}_{p,k} - h(f(\mathbf{x})_k) \bigr] \right).
```

这是一个定义在 6 维状态空间上的非常复杂的函数（经过非线性动力学传播 $`f`$ 和非线性测量 $`h`$）。它可能：
- 不是标准分布（高斯、伽马等）；
- 多峰、细长弯曲；
- 归一化常数未知（那个 $`c`$ 我们不知道）。

**没有现成的方法**能直接从这种分布中抽取独立样本。比如，你不能用 `randn` 或者 `mvnrnd` 来采样。

#### 2. 提议分布的“角色”：向导

M-H 算法采用了一个**妥协方案**：
1. **提议分布** $`q(\mathbf{x}' \mid \mathbf{x})`$ 是我们**自己设计**的、容易从中抽样的分布（例如高斯分布）。
2. 我们让链**游走**：从当前位置 $`\mathbf{x}`$，利用 $`q`$ 产生一个候选点 $`\mathbf{x}'`$。
3. 然后根据目标分布 $`p`$ 的值决定是否接受 $`\mathbf{x}'`$，使得最终马尔可夫链的稳态分布就是 $`p`$。

**通俗比喻**：
- 目标分布 $`p`$ 是一幅你从没见过的复杂地图（你想要在上面均匀走动，但地图上很多地方是禁区或者高坡）。
- 你不能直接按地图走，因为你不知道地图的全貌。
- 提议分布 $`q`$ 就是一个“盲人摸象”的试探步法：每次都从当前位置随机迈出一步（比如按高斯分布迈步）。
- 然后你根据迈到的新位置的地图高度（$`p`$ 的值）决定是留下来还是退回原地。
- 久而久之，你走出来的轨迹分布就符合地图的高度分布。

**没有提议分布，你就无法产生下一个候选点**。你只能僵在原地，或者只能随机瞎走（但瞎走也要有个规则，那就是提议分布）。

#### 3. 提议分布是必须的，即使对称

即使提议分布是对称的（$`q(x' \mid x) = q(x \mid x')`$），它仍然是一个**不可或缺的生成机制**。对称只是简化接受率计算，但并不免除我们需要一个“产生新样本的规则”这一事实。可以想象，如果你连提议分布都没有，M-H 算法就进行不下去了：无法从 $`p`$ 直接采样，也无法从任何别的分布生成候选，链就死掉了。

#### 4. 为什么不直接用重要性采样或其他方法？

重要性采样需要从某个**提议分布** $`q(x)`$ 中直接抽取独立样本，然后加权。但：
- 在高维非线性问题中，找一个好的 $`q(x)`$ 使得权重不发散非常困难。
- 而且每次都是独立样本，效率低。

M-H 的好处是通过**马尔可夫链**局部游走，利用之前样本的信息，更容易设计高效的局部提议分布（比如高斯随机游走）。

#### 5. 总结

> **提议分布是 M-H 算法用来“产生候选点”的工具**，因为我们不能直接从复杂的目标分布中抽样。没有它，马尔可夫链就无法移动。可以将“提议分布”理解为“**跳转规则**”或者“**建议分布**”——它只是提出一个建议，接受与否由目标分布 $`p`$ 的比值决定。
