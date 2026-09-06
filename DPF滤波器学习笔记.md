# 离散参数流滤波（DPF）学习笔记

> **论文**：AAS 25-582 — *Discrete Parameter Flow Filtering for Sparse Tracking of Objects in Cislunar Orbits*  
> **作者**：William N. Fife, Kyle J. DeMars, Gunner S. Fritsch  


---

## 第 0 章 阅读导引

### 0.1 DPF 是什么？

**离散参数流（Discrete Parameter Flow, DPF）** 是一种贝叶斯滤波**校正步**的实现方式：把传统“一步完成”的测量更新，拆成 **M 次小步增量更新**（伪时间步），每一步只融合一部分测量信息。

- **应用场景**：非线性系统、**稀疏观测**（如地月轨道目标跟踪，测量间隔长、先验不确定性大）。
- **核心表示**：高斯混合模型（GMM）描述状态 pdf；单高斯是 $L_x=1$ 的特例。
- **与单步 GMF 的关系**：式 (7)–(9) 本质上就是 **重复 M 次的 GMM 校正方程**；DPF 只是改变了每步使用的等效测量噪声 $P_{vv}/\Delta s_i$ 和迭代次数。

### 0.2 一句话对比

| 方法 | 校正步特点 |
|------|-----------|
| EKF / UKF | 单步更新；先验通常为单高斯 |
| EGMF / UGMF | 单步更新；先验为 GMM |
| DPF-EGMF / DPF-UGMF | M 步更新；每步等效噪声 $P_{vv}/\Delta s_i$ |
| ADPF-GMF | M 步 + 自适应 $\Delta s_i$（Algorithm 1） |

### 0.3 符号表

| 符号 | 含义 |
|------|------|
| $x_k \in \mathbb{R}^n$ | 时刻 $t_k$ 状态 |
| $z_k \in \mathbb{R}^m$ | 测量 |
| $f(\cdot),\, h(\cdot)$ | 动力学、观测模型 |
| $P_{ww},\, P_{vv}$ | 过程噪声、测量噪声协方差 |
| $L_x$ | GMM 分量数 |
| $M$ | 伪时间步数（校正迭代次数） |
| $s_m \in [0,1]$ | 伪时间点，如 $0=s_1 < s_2 < \cdots < s_{M+1}=1$ |
| $\Delta s_m = s_{m+1}-s_m$ | 第 $m$ 步伪时间增量，如 $\sum_m \Delta s_m = 1$ |
| $\ell=1,\ldots,L_x$ | GMM 分量索引 |
| $i=1,\ldots,M$ | 校正迭代步索引 |
| $w_x^{(\ell,i)},\, m_x^{(\ell,i)},\, P_{xx}^{(\ell,i)}$ | 权重、均值、协方差 |
| $k^{(\ell,i)},\, K^{(\ell,i)}$ | 权重增益、状态增益（式 8） |
| $m_h,\, P_{xh},\, P_{hh}$ | $h(x)$ 的均值及互/自协方差（式 9） |
| $p_g(y;\, m_y,\, P_{yy})$ | 多元高斯 pdf（论文记法 $p_g$） |

先验参数上标写 $(\ell)\text{-}$，后验写 $(\ell)\text{+}$（如 $w_x^{(\ell)\text{-}}$），与论文 $w_x^{(\ell)-}$ 含义相同。

---

## 第 1 章 为什么需要 DPF

### 1.1 传统单步校正

**式 (4)** — 标准贝叶斯校正（比例形式）：

$$
p(x_k \mid z_k) \propto p(z_k \mid x_k)\, p(x_k)
$$

完整 Bayes 形式：

$$
p(x_k \mid z_k) = \frac{1}{p(z_k)}\, p(z_k \mid x_k)\, p(x_k)
$$

工程上通常**一步**把整份测量 $z_k$ 融入先验。对线性高斯测量 $z = Hx + v$，这就是卡尔曼校正；对非线性 $h(x)$，需解析线性化（EKF）或统计线性化（UKF/求积）。

### 1.2 单步更新的问题

1. **线性化误差**：当 $h(x)$ 非线性时，在“大修正”下 Taylor 展开或 sigma 点近似误差大。
2. **GMM 过自信**：单步 GMF 中，许多分量一次吸收全部测量，尾部/低权重分量被大幅拉动，**NEES 显著偏大**（论文 2D 算例中 EGMF 中位 NEES $> 2$，极端可达 $\sim 7$）。
3. **稀疏观测场景**：地月轨道等长时间无测量，先验 spread 很大；一次 LOS 更新仍用单步 GMF 会**有偏且无法覆盖真实误差分布**。

### 1.3 DPF 的核心直觉

**不把全部似然一次乘上，而是分解为多个“小似然”逐步融合。**

**式 (5)** — 似然乘积分解。设 $0=s_1 < s_2 < \cdots < s_{M+1}=1$，且 $\sum_m \Delta s_m = 1$，其中 $\Delta s_m = s_{m+1}-s_m$：

$$
p(z_k \mid x_k) = \prod_{m=1}^{M} p^{\Delta s_m}(z_k \mid x_k)
$$

对高斯测量模型 $p_g(z_k;\, h(x_k),\, P_{vv})$，代入式 (5) 得 **式 (6)**：

$$
p(z_k \mid x_k) = \left|2\pi P_{vv}\right|^{-1/2} \prod_{m=1}^{M} \left|2\pi \frac{P_{vv}}{\Delta s_m}\right|^{1/2} p_g\left(z;\, h(x_k),\, \frac{P_{vv}}{\Delta s_m}\right)
$$

**要点**：

- $\Delta s_m > 0$ 时 $P_{vv}/\Delta s_m$ 非奇异（ $P_{vv}$ 非奇异前提下）。
- 第 $i$ 步迭代时，**有效测量噪声协方差**为 $P_{vv}/\Delta s_i$；当 $\Delta s_i$ 越小 → 该步测量越“弱” → 修正越小。

直觉上可记：

$$
R_{\text{step}} = \frac{P_{vv}}{\Delta s_i}
$$

每一步的线性化/统计线性化只处理“弱测量”，误差更小；当 $M$ 步累加后达到完整 Bayes 更新效果。

> **类比**：分步扩展卡尔曼 — 把一次大更新拆成 $M$ 次小卡尔曼校正；权重更新 $k^{(\ell,i)}$ 还考虑了 GMM 混合归一化（单高斯 Kalman 无此项）。

### 1.4 与相关工作的区别

| 方法 | 特点 |
|------|------|
| Progressive Bayes [3] | ODE 驱动近似分布参数 |
| Particle flow [4,5] | 连续同伦，粒子流动 |
| Zanetti [6] | 卡尔曼增益分块、重复线性化 |
| Michaelson [7,8] | 似然分割（高斯/粒子） |
| Craft [9] | **连续**参数流 + 数值积分 |
| **本文 DPF** | **有限步迭代**（可保证计算量）+ **自适应步长** Algorithm 1 |

与 Craft [9] 的区别：本文保留**有限迭代**（运行时间可预期），并给出 **Algorithm 1** 自适应选择 $\Delta s_m$。

---

## 第 2 章 贝叶斯滤波与 GMM 先验

### 2.1 系统模型 — 式 (1)

$$
x_k = f(x_{k-1}) + w_{k-1}
$$

$$
z_k = h(x_k) + v_k
$$

$w_{k-1},\, v_k$ 零均值，协方差分别为 $P_{ww},\, P_{vv}$。

### 2.2 预测步 — 式 (2)

$$
p(x_k \mid z_{1:k-1})
= \int_{\mathbb{R}^n} p(x_k \mid x_{k-1})\, p(x_{k-1} \mid z_{1:k-1})\, dx_{k-1}
$$

因为两个概率密度之间没有共轭关系，我们无法写出一个漂亮的解析解；同时，即使能写出来，结果分布也可能需要无限多的参数才能精确描述，这在工程上是不可行的，因此，通常选择一个参数化的密度模型（例如，高斯混合）并通过线性化、求积或蒙特卡洛技术近似预测方程。本文重点在**校正**；预测仍用常规 GMM 传播（线性化、求积或 Monte Carlo）。地月算例中均值与协方差平方根沿动力学 ODE 与 STM 传播（见论文第 10–11 页）。

### 2.3 先验 GMM — 式 (3)

$$
p(x_k)
= \sum_{\ell=1}^{L_x} w_x^{(\ell)\text{-}}\,
  p_g\left(x_k;\, m_x^{(\ell)\text{-}},\, P_{xx}^{(\ell)\text{-}}\right)
$$

- 上标 $(\ell)\text{-}$ 表示第 $\ell$ 个分量的**先验**参数（已吸收 $z_{1:k-1}$）。
- **单高斯特例**：当 $L_x=1$，此时 $w_x^{(1)\text{-}}=1$ → 退化为高斯 DPF（仍有式 7a 权重迭代）。

校正完成后，后验仍为 GMM：

$$
p(x_k \mid z_k)
= \sum_{\ell=1}^{L_x} w_x^{(\ell)\text{+}}\,
  p_g\left(x_k;\, m_x^{(\ell)\text{+}},\, P_{xx}^{(\ell)\text{+}}\right)
$$

---

## 第 3 章 离散参数流校正步（核心）

### 3.1 迭代初始化与输出

对第 $\ell$ 个 GMM 分量，当 $i=0$ 时：

$$
w_x^{(\ell,0)} = w_x^{(\ell)\text{-}}, \quad
m_x^{(\ell,0)} = m_x^{(\ell)\text{-}}, \quad
P_{xx}^{(\ell,0)} = P_{xx}^{(\ell)\text{-}}
$$

对 $1 \le i \le M$，依次应用式 (7)–(9)。迭代结束后：

$$
w_x^{(\ell)\text{+}} = w_x^{(\ell,M)}, \quad
m_x^{(\ell)\text{+}} = m_x^{(\ell,M)}, \quad
P_{xx}^{(\ell)\text{+}} = P_{xx}^{(\ell,M)}
$$

### 3.2 权重更新 — 式 (7a) 与 (8a)

$$
w_x^{(\ell,i)}
= \frac{k^{(\ell,i)}\, w_x^{(\ell,i-1)}}
       {\sum_{\ell'=1}^{L_x} k^{(\ell',i)}\, w_x^{(\ell',i-1)}}
$$

$$
k^{(\ell,i)}
= p_g\!\left(z;\, m_h^{(\ell,i-1)},\, P_{hh}^{(\ell,i-1)} + \frac{P_{vv}}{\Delta s_i}\right)
$$

**含义**：

- $k^{(\ell,i)}$ 是第 $i$ 步“部分似然”在第 $\ell$ 分量当前预测 $m_h$ 处的 pdf 值。
- 分母对所有分量归一化 → 混合权重重新分配。
- 与单高斯 Kalman 不同，**GMM 必须更新权重**。

### 3.3 均值更新 — 式 (7b) 与 (8b)

$$
m_x^{(\ell,i)}
= m_x^{(\ell,i-1)} + K^{(\ell,i)}\left(z - m_h^{(\ell,i-1)}\right)
$$

$$
K^{(\ell,i)}
= P_{xh}^{(\ell,i-1)} \left(P_{hh}^{(\ell,i-1)} + \frac{P_{vv}}{\Delta s_i}\right)^{-1}
$$

**含义**：标准 Kalman 增益形式，但：

- 使用**当前步**的 $P_{xh},\, P_{hh}$（在 $i-1$ 状态上计算）；
- 创新协方差中 $R = P_{vv}/\Delta s_i$ 随步长变化。

### 3.4 协方差更新 — 式 (7c)

$$
P_{xx}^{(\ell,i)} = P_{xx}^{(\ell,i-1)} - K^{(\ell,i)} P_{hh}^{(\ell,i-1)} \left(K^{(\ell,i)}\right)^{T} - K^{(\ell,i)} \frac{P_{vv}}{\Delta s_i} \left(K^{(\ell,i)}\right)^{T}
$$

**含义**：

- 第一项 $-K P_{hh} K^{T}$：与 Joseph 形式类似的**降级**更新（利用测量信息减不确定性）。
- 第二项 $-K (P_{vv}/\Delta s_i) K^{T}$：与分步噪声注入一致。
- 每步完成后 $P_{xx}$ 减小（在一致线性化下）。

### 3.5 测量统计量 — 式 (9)

期望关于 $p_g(x;\, m_x^{(\ell,\cdot)},\, P_{xx}^{(\ell,\cdot)})$ 取值：

$$
m_h^{(\ell,\cdot)} = \mathbb{E}\left[h(x)\right]
$$

$$
P_{xh}^{(\ell,\cdot)}
= \mathbb{E}\left[\left(x - m_x^{(\ell,\cdot)}\right)\left(h(x) - m_h^{(\ell,\cdot)}\right)^{T}\right]
$$

$$
P_{hh}^{(\ell,\cdot)}
= \mathbb{E}\left[\left(h(x) - m_h^{(\ell,\cdot)}\right)\left(h(x) - m_h^{(\ell,\cdot)}\right)^{T}\right]
$$

**关键**：每步 $i$ 都在**当前** $(m_x^{(\ell,i-1)},\, P_{xx}^{(\ell,i-1)})$ 下重新计算 (9)，再代入 (8)(7)。这是 DPF 与“冻结线性化点的一次 GMF”的本质区别。

### 3.6 单步迭代流程

```text
初始化 i=0：复制先验 w_x, m_x, P_xx
    ↓
┌─→ i ≤ M ?
│       │ 是
│       ↓
│   计算 (9)：m_h, P_xh, P_hh
│       ↓
│   计算 (8)：k, K
│       ↓
│   更新 (7)：w_x, m_x, P_xx
│       ↓
│   i ← i + 1 ──┘
│       │ 否
└───────┴→ 输出后验 GMM
```

---

## 第 4 章 滤波器变体

### 4.1 期望 (9) 的三种计算方式

| 名称 | 式 (9) 近似 | 说明 |
|------|-------------|------|
| **DPF-EGMF** | $m_h \approx h(m_x)$，这里 $P_{xh} \approx P_{xx} H_x^{T}$，这里 $P_{hh} \approx H_x P_{xx} H_x^{T}$ | 一阶 Taylor；这里 $H_x = \partial h / \partial x$ |
| **DPF-QGMF** | Gauss-Hermite 求积 | 通用求积框架 |
| **DPF-UGMF** | 无迹变换（UT） | $\alpha=0.1,\,\beta=2,\,\kappa=1$（论文 2D 算例） |

**DPF-EGMF** 显式写法：

$$
m_h^{(\ell,i-1)} \approx h\!\left(m_x^{(\ell,i-1)}\right)
$$

$$
P_{xh}^{(\ell,i-1)} \approx P_{xx}^{(\ell,i-1)} \left(H_x^{(\ell,i-1)}\right)^{T}
$$

$$
P_{hh}^{(\ell,i-1)} \approx H_x^{(\ell,i-1)} P_{xx}^{(\ell,i-1)} \left(H_x^{(\ell,i-1)}\right)^{T}
$$

**论文结论**：当 $M$ 足够大（如 40 步）时，EGMF 与 UGMF 性能接近——小步下解析与统计线性化差异变小。

### 4.2 平方根（SRF）实现 — 式 (13)

令 $S_{aa} S_{aa}^{T} = P_{aa}$。协方差平方根更新可写为（论文第 5 页）：

$$
S_{xx}^{(\ell,i)} S_{xx}^{(\ell,i)\,T} = S_{xx}^{(\ell,i-1)} S_{xx}^{(\ell,i-1)\,T} - K \left[ S_{hh} S_{hh}^{T} + \tfrac{1}{\Delta s_i} S_{vv} S_{vv}^{T} \right] K^{T}
$$

**线性化路径**：

$$
S_{hh}^{(\ell,i-1)} = H_x^{(\ell,i-1)} S_{xx}^{(\ell,i-1)}
$$

**求积路径**（含负权重 $w_c^{(\ell,j)}$）：按文献 [18] 分离正负权重，用 QR 与 Cholesky update/downdate 构造 $S_{hh}$（MATLAB 函数 `cholupdate`）。

**式 (13a)** — 先对创新协方差做 rank为 $m$ **update**：

$$
S_{zz}^{(\ell,i-1)} = \text{cholupdate}\left( S_{hh}^{(\ell,i-1)},\; \frac{S_{vv}}{\sqrt{\Delta s_i}} \right)
$$

**式 (13b)** — 再对状态协方差做 **update + downdate**：

$$
S_{xx}^{(\ell,i)} = \text{cholupdate}\left( S_{xx}^{(\ell,i-1)},\; K^{(\ell,i-1)} S_{zz}^{(\ell,i-1)} \right) \quad \text{(downdate)}
$$

式 (13b) 中 `cholupdate` 的第三个参数为减号标志（MATLAB 写法为 `cholupdate(..., '-')`），表示 Cholesky **downdate** 而非 update。

式 (13b) 依次做 rank为 $m$ 的 update 与 downdate（$m$ 为测量维数）。地月算例中三种滤波器均使用**线性化平方根**实现。

### 4.3 变体选择建议

| 需求 | 推荐 |
|------|------|
| 速度优先、弱非线性 | DPF-EGMF |
| 强非线性、可承受求积成本 | DPF-UGMF |
| 长期运行、数值稳定 | 平方根 DPF-EGMF / ADPF-GMF |
| 不确定步数 profile | ADPF-GMF |

---

## 第 5 章 自适应伪时间步长

### 5.1 固定步长的局限

- 等分 $\Delta s_m = 1/M$：实现简单，但 $\Delta s \to 0$ 时 $P_{vv}/\Delta s \to \infty$，有数值问题。
- 有时**单步**已与多步性能相近（先验不确定度与测量精度量级相当时）。
- 固定 $M=30$ 可能浪费计算（论文 ADPF 平均仅用 $\sim 16$ 步）。

### 5.2 自适应步长 — 式 (14)

对第 $\ell$ 分量、第 $m$ 次迭代：

$$
s_m^{(\ell)}
= \left(\frac{\left\|S_{vv}\right\|}{\left\|S_{hh}^{(\ell,m)}\right\|}\right)^{2}
$$

$\|\cdot\|$ 为谱范数。意图：使 $\|S_{hh}^{(\ell,i)}\|$ 与 $\|S_{vv}/\sqrt{s_i^{(\ell)}}\|$ **量级匹配**。

**Algorithm 1 约束**（概要）：

- 累计 $\bar{s} \leftarrow \bar{s} + s_m^{(\ell)}$，初始 $\bar{s}=0$；
- 若 $s_m^{(\ell)} > 1-\bar{s}$，则 $s_m^{(\ell)} \leftarrow 1-\bar{s}$；
- 最大步数 $M_{\max}$；
- 最小步长 $\varepsilon$：若 $s < \varepsilon$，则 $s \leftarrow \varepsilon$；
- 若 $m=M_{\max}$ 且 $\bar{s}<1$，最后一步取满剩余伪时间。

### 5.3 Algorithm 1（ADPF-GMF 步长选择）

```text
输入: S_hh^(ℓ,i), S_vv, M_max, ε
初始化: s̄ = 0, m = 1
while s̄ < 1 且 m < M_max:
    计算 s_m^(ℓ) ← 式 (14)
    if s_m^(ℓ) ≥ 1 − s̄:  s_m^(ℓ) ← 1 − s̄
    else if s < ε:        s ← ε
    if s_m^(ℓ) < 1 − s̄ 且 m = M_max:  s_m^(ℓ) ← 1 − s̄
    用式 (13) 执行该步更新
    m ← m+1,  s̄ ← s̄ + s_m^(ℓ)
对每个 GMM 分量 ℓ 重复
```

**命名**：采用 Algorithm 1 的 GMM 滤波器记为 **ADPF-GMF**。

**论文结果**：ELFO 算例中 ADPF-EGMF 与 DPF-EGMF（$M=30$ 线性步）统计性能几乎相同，但 ADPF **平均约 16 步**（约为固定步一半）。


---

## 第 6 章 总结与速查

### 6.1 三句话总结

1. DPF 将 Bayes 校正**同伦分解**为 $M$ 个伪时间步，每步使用 $P_{vv}/\Delta s_i$ 的弱测量更新。  
2. 式 (7)–(9) 即 **分步 GMF**；小步减轻线性化误差，保留 GMM 尾部质量。  
3. 平方根 (13) + Algorithm 1（ADPF）可在**约一半步数**下达到固定 DPF 性能。

### 6.2 公式速查（与论文编号一致）

| 编号 | 内容 |
|:----:|------|
| (1a)(1b) | 动力学 / 测量模型 |
| (2) | Chapman-Kolmogorov 预测 |
| (3) | 先验 GMM |
| (4) | Bayes 校正 |
| (5) | 似然乘积 $\prod_m p^{\Delta s_m}$ |
| (6) | 高斯似然等价形式 |
| (7a) | 权重 $w_x^{(\ell,i)}$ |
| (7b) | 均值 $m_x^{(\ell,i)}$ |
| (7c) | 协方差 $P_{xx}^{(\ell,i)}$ |
| (8a) | $k^{(\ell,i)}$ |
| (8b) | $K^{(\ell,i)}$ |
| (9a–c) | $m_h,\, P_{xh},\, P_{hh}$ |
| (13a)(13b) | SRF Cholesky 更新 |
| (14) | 自适应 $s_m^{(\ell)}$ |

### 6.3 实现检查清单

1. **预测**：得先验 GMM $\{w_x^{(\ell)\text{-}},\, m_x^{(\ell)\text{-}},\, P_{xx}^{(\ell)\text{-}}\}$。  
2. **选步长**：固定 $\Delta s_i$ 或 Algorithm 1。  
3. **初始化**：$i=0$ 复制先验。  
4. **for** $i=1,\ldots,M$（或直到 $\bar{s}=1$）：  
   - 算 (9)（EGMF/UGMF/求积）；  
   - 算 (8a)(8b)；  
   - 算 (7a)(7b)(7c)（或 SRF 式 13）；  
5. **输出** $i=M$ 为后验。  

### 6.4 整体算法流程

```text
预测 GMM 先验 → 选择 Δs_i 或 ADPF → M 步 DPF 迭代 (7)–(9) → 后验 GMM → 下一测量历元
```




