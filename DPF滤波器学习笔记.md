# 离散参数流滤波（DPF）学习笔记

> **论文**：AAS 25-582 — *Discrete Parameter Flow Filtering for Sparse Tracking of Objects in Cislunar Orbits*  
> **作者**：William N. Fife, Kyle J. DeMars, Gunner S. Fritsch  
> **整理依据**：论文正文 + 现有中文笔记（校正步公式讲解）
>
> **GitHub 阅读说明**：公式已按 MathJax 兼容写法排版（无 `\tag`、无公式内嵌套 `$`、符号表不用 LaTeX 表格）。推送后于 GitHub 网页打开 `.md` 即可渲染。

---

## 第 0 章 阅读导引

### 0.1 DPF 是什么？

**离散参数流（Discrete Parameter Flow, DPF）** 是一种贝叶斯滤波**校正步**的实现方式：把传统“一步完成”的测量更新，拆成 **M 次小步增量更新**（伪时间步），每一步只融合一部分测量信息。

- **应用场景**：非线性系统、**稀疏观测**（如地月轨道目标跟踪）。
- **核心表示**：GMM 描述状态 pdf；单高斯是 $L_x=1$ 的特例。
- **与单步 GMF 的关系**：式 (7)–(9) 即 **重复 M 次的 GMF 校正**；每步等效噪声 $P_{vv}/\Delta s_i$。

### 0.2 一句话对比

| 方法 | 校正步特点 |
|------|-----------|
| EKF / UKF | 单步；单高斯先验 |
| EGMF / UGMF | 单步；GMM 先验 |
| DPF-EGMF / DPF-UGMF | M 步；每步 $P_{vv}/\Delta s_i$ |
| ADPF-GMF | M 步 + 自适应 $\Delta s_i$ |

### 0.3 符号表

| 符号 | 含义 |
|------|------|
| $x_k$, $z_k$ | 状态、测量 |
| $P_{ww}$, $P_{vv}$ | 过程/测量噪声协方差 |
| $L_x$, $M$ | GMM 分量数、伪时间步数 |
| $\Delta s_m$ | 伪时间增量，$\sum_m \Delta s_m=1$ |
| $\ell$, $i$ | 分量索引、迭代步 |
| $w_x$, $m_x$, $P_{xx}$ | 权重、均值、协方差 |
| $k$, $K$ | 权重增益、状态增益 |
| $p_g$ | 多元高斯 pdf |

---

## 第 1 章 为什么需要 DPF

### 1.1 传统单步校正

**式 (4)**：

$$
p(x_k \mid z_k) \propto p(z_k \mid x_k)\, p(x_k)
$$

$$
p(x_k \mid z_k) = \frac{1}{p(z_k)}\, p(z_k \mid x_k)\, p(x_k)
$$

### 1.2 单步更新的问题

1. 非线性 $h(x)$ 大步更新 → 线性化误差大。  
2. 单步 GMF → NEES 偏大（2D 算例 EGMF 中位 NEES $>2$）。  
3. 稀疏观测 → 单步 GMF 有偏、误差带覆盖不足。

### 1.3 DPF 核心：似然分解

**式 (5)**：

$$
p(z_k \mid x_k) = \prod_{m=1}^{M} p^{\Delta s_{m}}(z_k \mid x_k)
$$

**式 (6)**：

$$
p(z_k \mid x_k)
= |2\pi P_{vv}|^{-1/2}
  \prod_{m=1}^{M} \left|2\pi \frac{P_{vv}}{\Delta s_m}\right|^{1/2}
  p_g\!\left(z;\, h(x_k),\, \frac{P_{vv}}{\Delta s_m}\right)
$$

第 $i$ 步等效噪声：$R_{\mathrm{step}} = P_{vv}/\Delta s_i$。

### 1.4 相关工作

Progressive Bayes、Particle flow、Zanetti 增益分割、Craft 连续流等；**本文**：有限步 DPF + Algorithm 1 自适应步长。

---

## 第 2 章 贝叶斯滤波与 GMM 先验

**式 (1a)(1b)**：

$$
x_k = f(x_{k-1}) + w_{k-1}, \qquad z_k = h(x_k) + v_k
$$

**式 (2)**：

$$
p(x_k \mid z_{1:k-1})
= \int_{\mathbb{R}^n} p(x_k \mid x_{k-1})\, p(x_{k-1} \mid z_{1:k-1})\, dx_{k-1}
$$

**式 (3)**：

$$
p(x_k)
= \sum_{\ell=1}^{L_x} w_x^{(\ell)\text{-}}\,
  p_g\!\left(x_k;\, m_x^{(\ell)\text{-}},\, P_{xx}^{(\ell)\text{-}}\right)
$$

后验 GMM 将 $(\ell)\text{-}$ 换为 $(\ell)\text{+}$。$L_x=1$ 时退化为高斯 DPF。

---

## 第 3 章 离散参数流校正步

初始化 $i=0$：$w_x^{(\ell,0)}=w_x^{(\ell)\text{-}}$，$m_x$、$P_{xx}$ 同理。

**式 (7a)**：

$$
w_x^{(\ell,i)}
= \frac{k^{(\ell,i)}\, w_x^{(\ell,i-1)}}
       {\sum_{\ell'=1}^{L_x} k^{(\ell',i)}\, w_x^{(\ell',i-1)}}
$$

**式 (8a)**：

$$
k^{(\ell,i)}
= p_g\!\left(z;\, m_h^{(\ell,i-1)},\, P_{hh}^{(\ell,i-1)} + \frac{P_{vv}}{\Delta s_i}\right)
$$

**式 (7b)**：

$$
m_x^{(\ell,i)}
= m_x^{(\ell,i-1)} + K^{(\ell,i)}\left(z - m_h^{(\ell,i-1)}\right)
$$

**式 (8b)**：

$$
K^{(\ell,i)}
= P_{xh}^{(\ell,i-1)} \left(P_{hh}^{(\ell,i-1)} + \frac{P_{vv}}{\Delta s_i}\right)^{-1}
$$

**式 (7c)**：

$$
\begin{aligned}
P_{xx}^{(\ell,i)}
&= P_{xx}^{(\ell,i-1)}
 - K^{(\ell,i)} P_{hh}^{(\ell,i-1)} \left(K^{(\ell,i)}\right)^{T} \\
&\quad - K^{(\ell,i)} \frac{P_{vv}}{\Delta s_i} \left(K^{(\ell,i)}\right)^{T}
\end{aligned}
$$

**式 (9a)(9b)(9c)**：

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

每步 $i$ 重算 (9) 再代入 (8)(7)。输出 $i=M$ 为后验。

---

## 第 4 章 滤波器变体

**DPF-EGMF**：$m_h \approx h(m_x)$，$P_{xh} \approx P_{xx} H_x^{T}$，$P_{hh} \approx H_x P_{xx} H_x^{T}$。

**DPF-UGMF**：无迹变换求 (9)。$M$ 足够大时 EGMF 与 UGMF 接近。

**式 (13a)**：

$$
S_{zz}^{(\ell,i-1)}
= \mathrm{cholupdate}\!\left\{S_{hh}^{(\ell,i-1)},\; \frac{1}{\sqrt{\Delta s_i}}\, S_{vv}\right\}
$$

**式 (13b)**（downdate 用 `\text{downdate}` 标记，对应 MATLAB `cholupdate(...,'-'`）：

$$
S_{xx}^{(\ell,i)}
= \mathrm{cholupdate}\!\left\{S_{xx}^{(\ell,i-1)},\; K^{(\ell,i-1)} S_{zz}^{(\ell,i-1)},\; \text{downdate}\right\}
$$

---

## 第 5 章 自适应步长

**式 (14)**：

$$
s_m^{(\ell)}
= \left(\frac{\left\|S_{vv}\right\|}{\left\|S_{hh}^{(\ell,m)}\right\|}\right)^{2}
$$

Algorithm 1 → **ADPF-GMF**（ELFO 算例平均约 16 步 vs 固定 30 步）。

---

## 第 6 章 算例

**NEES**：$d = \frac{1}{n}(x-m_x^+)^{T}(P_{xx}^+)^{-1}(x-m_x^+)$，一致时 $\mathbb{E}[d]=n$。

- **2D**：DPF 误差 ↓50%，NEES 接近 1（归一化）。  
- **ELFO**：DPF/ADPF 无偏；3 轨道周期后 RSS $\approx 120\,\mathrm{m}$。

---

## 第 7 章 速查

| 编号 | 内容 |
|:----:|------|
| (1)–(3) | 模型、预测、先验 GMM |
| (4)–(6) | Bayes、似然分解 |
| (7)–(9) | DPF 迭代核心 |
| (13) | 平方根更新 |
| (14) | 自适应步长 |

**实现**：预测 → 选 $\Delta s_i$ → for $i=1..M$： (9)→(8)→(7) → 后验。

---

## 附录

- 来源：AAS 25-582 论文 + 原中文笔记  
- 先验/后验上标：$w_x^{(\ell)\text{-}}$、$w_x^{(\ell)\text{+}}$（GitHub 安全写法）  
- 权重增益：$k^{(\ell,i)}$（论文式 8a）

### 本次 GitHub 渲染修复摘要

| 问题 | 处理 |
|------|------|
| `\tag{7a}` 等 | 改为正文 **式 (7a)** 标题 |
| `w_x^{(\ell)-}` 上标 | 改为 `w_x^{(\ell)\text{-}}` |
| 式 (13b) 内嵌 `$-$` | 改为 `\text{downdate}` |
| `\boxed{aligned}` + 中文 | 改为文字步骤列表 |
| 7c 换行被当成 Markdown 列表 | 用 `\begin{aligned}...\end{aligned}` |
| 9b/9c 缺 `}` | 补全 `\mathbb{E}\left[...\right]` |
| 符号表复杂 LaTeX | 表格用简写，详式见正文 |

请重新 `git add`、`commit`、`push` 后在 GitHub 刷新页面查看。
