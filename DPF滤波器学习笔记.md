# 离散参数流滤波（DPF）学习笔记

> **论文**：AAS 25-582 — *Discrete Parameter Flow Filtering for Sparse Tracking of Objects in Cislunar Orbits*  
> **作者**：William N. Fife, Kyle J. DeMars, Gunner S. Fritsch  
> **整理依据**：论文正文 + 现有中文笔记（校正步公式讲解）
>
> **公式阅读**：正文公式与论文排版一致（LaTeX）。在 Cursor / VS Code 中按 `Ctrl+Shift+V` 打开 **Markdown 预览**；或使用 Typora、Obsidian 查看。

---

## 第 0 章 阅读导引

### 0.1 DPF 是什么？

**离散参数流（Discrete Parameter Flow, DPF）** 是一种贝叶斯滤波**校正步**的实现方式：把传统“一步完成”的测量更新，拆成 **M 次小步增量更新**（伪时间步），每一步只融合一部分测量信息。

- **应用场景**：非线性系统、**稀疏观测**（如地月轨道目标跟踪，测量间隔长、先验不确定性大）。
- **核心表示**：高斯混合模型（GMM）描述状态 pdf；单高斯是 $L_x=1$ 的特例。
- **与单步 GMF 的关系**：式 (7)–(9) 本质上就是 **重复 M 次的 GMF 校正方程**；DPF 只是改变了每步使用的等效测量噪声 $P_{vv}/\Delta s_i$ 和迭代次数。

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
| $s_m \in [0,1]$ | 伪时间点，$0=s_1 < s_2 < \cdots < s_{M+1}=1$ |
| $\Delta s_m = s_{m+1}-s_m$ | 第 $m$ 步伪时间增量，$\sum_m \Delta s_m = 1$ |
| $\ell=1,\ldots,L_x$ | GMM 分量索引 |
| $i=1,\ldots,M$ | 校正迭代步索引 |
| $w_x^{(\ell,i)},\, m_x^{(\ell,i)},\, P_{xx}^{(\ell,i)}$ | 权重、均值、协方差 |
| $k^{(\ell,i)},\, K^{(\ell,i)}$ | 权重增益、状态增益（式 8） |
| $m_h,\, P_{xh},\, P_{hh}$ | $h(x)$ 的均值及互/自协方差（式 9） |
| $p_g(y;\, m_y,\, P_{yy})$ | 多元高斯 pdf |

---

## 第 1 章 为什么需要 DPF

### 1.1 传统单步校正

标准贝叶斯校正（论文式 4，比例形式）：

$$
p(x_k \mid z_k) \propto p(z_k \mid x_k)\, p(x_k)
\tag{4}
$$

完整形式：

$$
p(x_k \mid z_k) = \frac{1}{p(z_k)}\, p(z_k \mid x_k)\, p(x_k)
$$

工程上通常**一步**把整份测量 $z_k$ 融入先验。对线性高斯测量 $z = Hx + v$，这就是卡尔曼校正；对非线性 $h(x)$，需解析线性化（EKF）或统计线性化（UKF/求积）。

### 1.2 单步更新的问题

1. **线性化误差**：$h(x)$ 非线性时，在“大修正”下 Taylor 展开或 sigma 点近似误差大。
2. **GMM 过自信**：单步 GMF 中，许多分量一次吸收全部测量，尾部/低权重分量被大幅拉动，NEES 显著偏大（论文 2D 算例中 EGMF 中位 NEES $> 2$，极端可达 $\sim 7$）。
3. **稀疏观测场景**：地月轨道等长时间无测量，先验 spread 很大；一次 LOS 更新仍用单步 GMF 会**有偏且无法覆盖真实误差分布**。

### 1.3 DPF 的核心直觉

**似然乘积分解**（论文式 5）。设 $0=s_1 < s_2 < \cdots < s_{M+1}=1$，且 $\sum_m \Delta s_m = 1$，其中 $\Delta s_m = s_{m+1}-s_m$：

$$
p(z_k \mid x_k) = \prod_{m=1}^{M} p^{\Delta s_m}(z_k \mid x_k)
\tag{5}
$$

对高斯测量模型 $p_g(z_k;\, h(x_k),\, P_{vv})$，代入式 (5) 得论文式 (6)：

$$
p(z_k \mid x_k)
= |2\pi P_{vv}|^{-1/2}
  \prod_{m=1}^{M} |2\pi P_{vv}/\Delta s_m|^{1/2}
  \times p_g\!\bigl(z;\, h(x_k),\, P_{vv}/\Delta s_m\bigr)
\tag{6}
$$

第 $i$ 步迭代时，等效测量噪声协方差为 $P_{vv}/\Delta s_i$；$\Delta s_i$ 越小，该步修正越弱。直觉上可记：

$$
R_{\mathrm{step}} = \frac{P_{vv}}{\Delta s_i}
$$

每一步的线性化/统计线性化只处理“弱测量”；$M$ 步累加后达到完整 Bayes 更新效果。

> **类比**：分步扩展卡尔曼 — 把一次大更新拆成 $M$ 次小卡尔曼校正；权重更新 $k^{(\ell,i)}$ 还考虑了 GMM 混合归一化。

### 1.4 与相关工作的区别

| 方法 | 特点 |
|------|------|
| Progressive Bayes [3] | ODE 驱动近似分布参数 |
| Particle flow [4,5] | 连续同伦，粒子流动 |
| Zanetti [6] | 卡尔曼增益分块、重复线性化 |
| Michaelson [7,8] | 似然分割（高斯/粒子） |
| Craft [9] | 连续参数流 + 数值积分 |
| **本文 DPF** | 有限步迭代 + 自适应步长 Algorithm 1 |

---

## 第 2 章 贝叶斯滤波与 GMM 先验

### 2.1 系统模型 — 式 (1)

$$
x_k = f(x_{k-1}) + w_{k-1}
\tag{1a}
$$

$$
z_k = h(x_k) + v_k
\tag{1b}
$$

$w_{k-1},\, v_k$ 零均值，协方差分别为 $P_{ww},\, P_{vv}$。

### 2.2 预测步 — 式 (2)

$$
p(x_k \mid z_{1:k-1})
= \int_{\mathbb{R}^n} p(x_k \mid x_{k-1})\, p(x_{k-1} \mid z_{1:k-1})\, dx_{k-1}
\tag{2}
$$

本文重点在**校正**；预测仍用常规 GMM 传播（线性化、求积或 Monte Carlo）。

### 2.3 先验 GMM — 式 (3)

$$
p(x_k)
= \sum_{\ell=1}^{L_x} w_x^{(\ell)-}\,
  p_g\!\bigl(x_k;\, m_x^{(\ell)-},\, P_{xx}^{(\ell)-}\bigr)
\tag{3}
$$

- 上标 $(\ell)-$ 表示第 $\ell$ 个分量的**先验**参数（已吸收 $z_{1:k-1}$）。
- **单高斯特例**：$L_x=1$，$w_x^{(1)-}=1$ → 退化为高斯 DPF（仍有式 7a 权重迭代）。

校正完成后，后验仍为 GMM：

$$
p(x_k \mid z_k)
= \sum_{\ell=1}^{L_x} w_x^{(\ell)+}\,
  p_g\!\bigl(x_k;\, m_x^{(\ell)+},\, P_{xx}^{(\ell)+}\bigr)
$$

---

## 第 3 章 离散参数流校正步（核心）

### 3.1 迭代初始化与输出

对第 $\ell$ 个 GMM 分量，$i=0$ 时：

$$
w_x^{(\ell,0)} = w_x^{(\ell)-}, \quad
m_x^{(\ell,0)} = m_x^{(\ell)-}, \quad
P_{xx}^{(\ell,0)} = P_{xx}^{(\ell)-}
$$

对 $1 \le i \le M$ 依次应用式 (7)–(9)。迭代结束后：

$$
w_x^{(\ell)+} = w_x^{(\ell,M)}, \quad
m_x^{(\ell)+} = m_x^{(\ell,M)}, \quad
P_{xx}^{(\ell)+} = P_{xx}^{(\ell,M)}
$$

### 3.2 权重更新 — 式 (7a) 与 (8a)

$$
w_x^{(\ell,i)}
= \frac{k^{(\ell,i)}\, w_x^{(\ell,i-1)}}
       {\displaystyle\sum_{\ell'=1}^{L_x} k^{(\ell',i)}\, w_x^{(\ell',i-1)}}
\tag{7a}
$$

$$
k^{(\ell,i)}
= p_g\!\Bigl(z;\, m_h^{(\ell,i-1)},\, P_{hh}^{(\ell,i-1)} + P_{vv}/\Delta s_i\Bigr)
\tag{8a}
$$

**含义**：$k^{(\ell,i)}$ 为第 $i$ 步部分似然在分量 $\ell$ 当前预测处的 pdf 值；分母归一化混合权重。**GMM 必须更新权重**，单高斯 Kalman 无此项。

### 3.3 均值更新 — 式 (7b) 与 (8b)

$$
m_x^{(\ell,i)}
= m_x^{(\ell,i-1)} + K^{(\ell,i)}\,\bigl(z - m_h^{(\ell,i-1)}\bigr)
\tag{7b}
$$

$$
K^{(\ell,i)}
= P_{xh}^{(\ell,i-1)} \left(P_{hh}^{(\ell,i-1)} + P_{vv}/\Delta s_i\right)^{-1}
\tag{8b}
$$

**含义**：标准 Kalman 增益；创新协方差含 $P_{vv}/\Delta s_i$；$P_{xh},\, P_{hh}$ 每步在 $(m_x^{(\ell,i-1)},\, P_{xx}^{(\ell,i-1)})$ 上重算。

### 3.4 协方差更新 — 式 (7c)

$$
P_{xx}^{(\ell,i)}
= P_{xx}^{(\ell,i-1)}
  - K^{(\ell,i)} P_{hh}^{(\ell,i-1)} \bigl(K^{(\ell,i)}\bigr)^{\!T}
  - K^{(\ell,i)} \frac{P_{vv}}{\Delta s_i} \bigl(K^{(\ell,i)}\bigr)^{\!T}
\tag{7c}
$$

**含义**：第一项为测量信息导致的协方差缩减；第二项与分步噪声注入 $P_{vv}/\Delta s_i$ 一致。

### 3.5 测量统计量 — 式 (9)

期望关于 $p_g(x;\, m_x^{(\ell,\cdot)},\, P_{xx}^{(\ell,\cdot)})$ 取值：

$$
m_h^{(\ell,\cdot)} = \mathbb{E}\{h(x)\}
\tag{9a}
$$

$$
P_{xh}^{(\ell,\cdot)}
= \mathbb{E}\bigl\{(x - m_x^{(\ell,\cdot)})\,(h(x) - m_h^{(\ell,\cdot)})^{\!T}\bigr\}
\tag{9b}
$$

$$
P_{hh}^{(\ell,\cdot)}
= \mathbb{E}\bigl\{(h(x) - m_h^{(\ell,\cdot)})\,(h(x) - m_h^{(\ell,\cdot)})^{\!T}\bigr\}
\tag{9c}
$$

**关键**：每步 $i$ 在**当前** $(m_x^{(\ell,i-1)},\, P_{xx}^{(\ell,i-1)})$ 下重算 (9)，再代入 (8)(7) — 区别于“冻结线性化点”的单步 GMF。

### 3.6 单步迭代流程

$$
\boxed{
\begin{aligned}
&\text{初始化 } i=0:\; w_x^{(\ell,0)},\, m_x^{(\ell,0)},\, P_{xx}^{(\ell,0)} \leftarrow \text{先验} \\
&\textbf{for } i=1,\ldots,M: \\
&\quad \text{计算 (9)} \;\Rightarrow\; m_h,\, P_{xh},\, P_{hh} \\
&\quad \text{计算 (8)} \;\Rightarrow\; k,\, K \\
&\quad \text{更新 (7)} \;\Rightarrow\; w_x,\, m_x,\, P_{xx} \\
&\text{输出 } w_x^{(\ell,+)},\, m_x^{(\ell,+)},\, P_{xx}^{(\ell,+)}
\end{aligned}
}
$$

---

## 第 4 章 滤波器变体

### 4.1 期望 (9) 的近似 — DPF-EGMF / QGMF / UGMF

**DPF-EGMF**（一阶 Taylor，论文第 4 页）：

$$
m_h^{(\ell,i-1)} \approx h\!\bigl(m_x^{(\ell,i-1)}\bigr)
$$

$$
P_{xh}^{(\ell,i-1)} \approx P_{xx}^{(\ell,i-1)} \bigl(H_x^{(\ell,i-1)}\bigr)^{\!T}
$$

$$
P_{hh}^{(\ell,i-1)} \approx H_x^{(\ell,i-1)} P_{xx}^{(\ell,i-1)} \bigl(H_x^{(\ell,i-1)}\bigr)^{\!T}
$$

其中 $H_x^{(\ell,i-1)} = \partial h / \partial x \big|_{m_x^{(\ell,i-1)}}$。

**DPF-QGMF / DPF-UGMF**：用 Gauss-Hermite 求积或无迹变换（UT，$\alpha=0.1,\,\beta=2,\,\kappa=1$）对式 (9) 求期望。$M$ 足够大（如 40 步）时，EGMF 与 UGMF 性能接近。

### 4.2 平方根（SRF）实现 — 式 (13)

令 $S_{aa} S_{aa}^{\!T} = P_{aa}$。线性化路径：

$$
S_{hh}^{(\ell,i-1)} = H_x^{(\ell,i-1)} S_{xx}^{(\ell,i-1)}
$$

求积路径（含负权重 $w_c^{(\ell,j)}$）见文献 [18]（QR + Cholesky update/downdate）。

**论文式 (13a)(13b)**：

$$
S_{zz}^{(\ell,i-1)}
= \mathrm{cholupdate}\!\left\{S_{hh}^{(\ell,i-1)},\; \frac{1}{\sqrt{\Delta s_i}}\, S_{vv}\right\}
\tag{13a}
$$

$$
S_{xx}^{(\ell,i)}
= \mathrm{cholupdate}\!\left\{S_{xx}^{(\ell,i-1)},\; K^{(\ell,i-1)} S_{zz}^{(\ell,i-1)},\; \text{``$-$''}\right\}
\tag{13b}
$$

式 (13b) 依次做 rank-$m$ Cholesky **update** 与 **downdate**（$m$ 为测量维数）。

### 4.3 变体选择

| 需求 | 推荐 |
|------|------|
| 速度优先、弱非线性 | DPF-EGMF |
| 强非线性 | DPF-UGMF |
| 数值稳定、长期运行 | 平方根 DPF-EGMF / ADPF-GMF |
| 不确定步长 profile | ADPF-GMF |

---

## 第 5 章 自适应伪时间步长

### 5.1 固定步长的局限

- 等分 $\Delta s_m = 1/M$：$\Delta s \to 0$ 时 $P_{vv}/\Delta s \to \infty$，有数值问题。
- 有时单步更新已与多步性能相近。
- 固定 $M=30$ 可能浪费计算（ADPF 平均约 16 步）。

### 5.2 自适应步长 — 式 (14)

对第 $\ell$ 分量、第 $m$ 次迭代：

$$
s_m^{(\ell)}
= \left(\frac{\|S_{vv}\|}{\|S_{hh}^{(\ell,m)}\|}\right)^{\!2}
\tag{14}
$$

$\|\cdot\|$ 为谱范数；使 $\|S_{hh}^{(\ell,i)}\|$ 与 $\|S_{vv}/\sqrt{s_i^{(\ell)}}\|$ 量级匹配。

**Algorithm 1 约束**（概要）：

- 累计 $\bar{s} \leftarrow \bar{s} + s_m^{(\ell)}$，初始 $\bar{s}=0$；
- 若 $s_m^{(\ell)} > 1-\bar{s}$，则 $s_m^{(\ell)} \leftarrow 1-\bar{s}$；
- 最大步数 $M_{\max}$；最小步长 $\varepsilon$；
- 若 $m=M_{\max}$ 且 $\bar{s}<1$，最后一步取满剩余伪时间。

采用 Algorithm 1 的 GMM 滤波器记为 **ADPF-GMF**。ELFO 算例中 ADPF 与 DPF-EGMF（$M=30$）性能相近，但平均约 **16 步**。

---

## 第 6 章 算例与性能指标

### 6.1 评价指标

**后验误差范数**：

$$
\|e_x^+\| = \|x - m_x^+\|
$$

**归一化估计误差平方（NEES）**：

$$
d = \frac{1}{n}\,(x - m_x^+)^{\!T} (P_{xx}^+)^{-1} (x - m_x^+)
$$

- 一致滤波器：$\mathbb{E}[d] = n$（$n$ 为状态维数）。
- $d/n > 1$：过自信；$d/n < 1$：保守。

### 6.2 二维 GMM 演示

- 状态 $x=[x,\,y]^{\!T}$；从先验 GMM **尾部** $x_{\mathrm{tail}}=[44,\,10]^{\!T}$ 采样 1000 组高精度测量；
- $P_{vv} = \mathrm{diag}(2.5\times10^{-5},\, 0.5^2\ \mathrm{arcsec}^2)$；
- DPF：$M=40$，三次伪时间规则；
- **结论**：DPF-EGMF/UGMF 相对 EGMF/UGMF，最坏情况误差约降 50%，中位 NEES 接近一致（$\sim 1$ 归一化）。

### 6.3 地月 ELFO 跟踪

- 27 分量 GMM；月心 LOS；传播 1–3 轨道周期后单次测量；
- EGMF 单步：有偏、过自信；DPF/ADPF：无偏，$\pm 3\sigma$ 与 MC 一致；
- 3 轨道周期后，多步滤波中位 RSS 位置误差 $\approx 120\,\mathrm{m}$。

---

## 第 7 章 总结与速查

### 7.1 三句话总结

1. DPF 将 Bayes 校正分解为 $M$ 个伪时间步，每步使用 $P_{vv}/\Delta s_i$ 的弱测量更新。  
2. 式 (7)–(9) 即分步 GMF；小步减轻线性化误差，保留 GMM 尾部。  
3. 平方根 (13) + ADPF 可在约一半步数下达到固定 DPF 性能。

### 7.2 公式速查（与论文编号一致）

| 编号 | 公式 |
|:----:|------|
| (1a) | $x_k = f(x_{k-1}) + w_{k-1}$ |
| (1b) | $z_k = h(x_k) + v_k$ |
| (2) | Chapman-Kolmogorov 预测 |
| (3) | 先验 GMM |
| (4) | Bayes 校正 |
| (5) | 似然乘积 $\prod_m p^{\Delta s_m}$ |
| (6) | 高斯似然等价形式 |
| (7a) | $w_x^{(\ell,i)}$ 更新 |
| (7b) | $m_x^{(\ell,i)}$ 更新 |
| (7c) | $P_{xx}^{(\ell,i)}$ 更新 |
| (8a) | $k^{(\ell,i)}$ |
| (8b) | $K^{(\ell,i)}$ |
| (9a–c) | $m_h,\, P_{xh},\, P_{hh}$ |
| (13a)(13b) | SRF Cholesky 更新 |
| (14) | 自适应 $s_m^{(\ell)}$ |

### 7.3 实现检查清单

1. 预测 → 先验 GMM $\{w_x^{(\ell)-},\, m_x^{(\ell)-},\, P_{xx}^{(\ell)-}\}$  
2. 选 $\Delta s_i$（固定或 Algorithm 1）  
3. $i=0$ 初始化  
4. **for** $i=1,\ldots,M$：算 (9) → (8) → (7) 或 (13)  
5. 输出 $i=M$ 为后验  

### 7.4 参考文献（精选）

- [12] Alspach & Sorenson — GMM 非线性滤波  
- [14] DeMars & Fife — 式 (7) 证明（Appendix A）  
- [16] Julier & Uhlmann — 无迹变换  
- [17] Carpenter & D'Souza — 导航滤波最佳实践  
- [18] McCabe & DeMars — 负权重平方根 Consider 滤波  

---

## 附录

### 主要来源

- `DISCRETE PARAMETER FLOWFILTERING FOR SPARSE.pdf`（AAS 25-582）
- 原 Word 笔记及 DeepSeek 校正步讲解

公式符号、编号与论文正文一致；权重增益记为 $k^{(\ell,i)}$（论文式 8a，非 $\kappa$）。
