# 自适应高斯混合交互多模型滤波（AGMIMM）学习笔记

> **论文**：*Adaptive Gaussian Mixture Filtering for Multi-sensor Maneuvering Cislunar Space Object Tracking*  
> **期刊**：The Journal of the Astronautical Sciences (2025) 72:2  
> **DOI**：[10.1007/s40295-024-00478-z](https://doi.org/10.1007/s40295-024-00478-z)  
> **作者**：John L. Iannamorelli, Keith A. LeGrand（Purdue University）  
> **整理依据**：论文 PDF + `AGMIMM.docx` 笔记逻辑；各节 **【文字说明】** 为 docx 式中文讲解（公式编号对齐论文 Sect. 2–5）
>
> **阅读说明**：LaTeX 已针对 **GitHub 数学渲染**优化：display 公式一律单行、禁用 `\operatorname` / `\mathbb{}`、先验/后验上标用 `\text{-}` / `\text{+}`；正文表格与列表尽量少用行内 `$...$`。

---

## 第 0 章 阅读导引

### 0.1 AGMIMM 是什么？

**自适应高斯混合交互多模型（Adaptive Gaussian Mixture Interacting Multiple Model, AGMIMM）** 是一种面向 **非线性跳跃马尔可夫系统（JMS）** 的贝叶斯滤波器，专门用于 **地月空间非合作机动 CSO** 的 **仅角度（angles-only）多传感器跟踪**。

核心思想可以拆成三层：

1. **IMM 框架**：目标模式（弹道 / 机动）未知，用马尔可夫链描述随机切换；各模式维护条件 pdf。
2. **GMM 表示**：每个模式下的 pdf 用高斯混合（GM）近似，以刻画混沌 CR3BP 动力学产生的 **强非高斯** 不确定性。
3. **自适应 AGM**：在动力学非线性、测量非线性、FoV 边界、负信息（未检测）和 Jacobi 约束等区域 **动态分裂** 混合分量；在模式交互混合步用 **GMRC 聚类合并** 控制计算量。

与现有方法相比，AGMIMM 的两项关键创新是：

- 相对一般 AGM 滤波器：**额外适应随机模式切换**（JMS/IMM 结构）。
- 相对 IMM/GMIMM：**多准则分裂 + 基于聚类的混合后验合并**，而非单高斯或按权重裁剪合并。

**【文字说明】** 可以把 AGMIMM 理解成在 **三条轴** 上同时做自适应：(1) **模式轴**——IMM 交互与模式概率更新；(2) **表示轴**——GMM 分量数通过分裂增、通过 GMRC 减；(3) **信息轴**——正检测、负检测、Jacobi/碰撞约束分别对应不同分裂或剪枝准则。一般 AGM 文献多关注 (2)(3) 中的动力学与测量非线性；本文 **首次** 在 JMS/IMM 框架下系统整合 GMRC 与多准则 AGM，并应用于地月仅角度多传感器跟踪。

### 0.2 一句话对比

| 方法 | 状态表示 | 模式切换 | 负信息 / FoV | 非高斯适应 |
|------|----------|----------|--------------|------------|
| EKF / SRUKF | 单高斯 | 通常单模型 | 难系统处理 | 无 |
| IMM | 各模式单高斯 | 马尔可夫 IMM | 未专门处理 | 弱 |
| GMIMM | 各模式 GMM | IMM + GMM | 未专门处理 | 部分（固定分量） |
| **AGMIMM** | 各模式 GMM | IMM + **GMRC 合并** | **RFS + 递归分裂** | **多准则 AGM 分裂** |

### 0.3 符号表

| 符号 | 含义 |
|------|------|
| $x_k \in \text{R}^{n_x}$ | 时刻 $t_k$ 运动学状态（位置+速度，共 6 维） |
| $\tilde{x}_k = [x_k^{\top}, \tau_k]^{\top}$ | 联合状态（运动学 + 模式） |
| $\tau_k \in \mathcal{M}=\{1,\ldots,M\}$ | 目标模式；本文 $M=2$（弹道/机动） |
| $\pi_{ij}$ | 模式转移概率 $\Pr(\tau_k=j \mid \tau_{k-1}=i)$ |
| $\mu_k^{(j)}$ | 时刻 $t_k$ 模式 $j$ 的后验概率 |
| $\bar{\mu}_{k|k-1}^{(j)}$ | 时刻 $t_k$ 模式 $j$ 的先验（混合后）概率 |
| $\omega^{(\ell,j)}, m^{(\ell,j)}, P^{(\ell,j)}$ | 模式 $j$ 下第 $\ell$ 个 GM 分量的权重、均值、协方差 |
| $L^{(j)}$ | 模式 $j$ 的 GM 分量数；合并目标数 $L \lt L_{\max}$ |
| $f(\cdot)$ | CR3BP 解流（离散动力学） |
| $h^{(o)}(\cdot)$ | 观测星 $o$ 的角度测量模型 |
| $Z_k^{(o)}$ | 观测星 $o$ 的 Bernoulli RFS 测量（空集或单点） |
| $p_D^{(o)}(x; S^{(o)})$ | 状态相关的检测概率 |
| $C(r,v)$ | Jacobi 常数（弹道积分量） |
| $[\text{LU}], [\text{TU}]$ | CR3BP 归一化长度/时间单位 |

**上标约定**：混合后、预测前参数常记 $(\cdot)_{k-1|k-1}$；预测后验记 $(\cdot)_{k|k-1}$；测量更新后记 $(\cdot)_k$ 或 $(\cdot)^{\text{+}}$。

---

## 第 1 章 问题 formulation 与动机

### 1.1 为什么地月 CSO 跟踪特别难？

**直觉链条**：

1. **动力学**：CR3BP 是强非线性、混沌系统 → 即使目标弹道飞行，长时间传播后 pdf 也 **高度非高斯**。
2. **观测**：地月空间监视受月球遮挡、光照、月球排除角限制 → **长观测间隙** 常见；间隙内不确定性快速增长。
3. **机动**：非合作目标点火时刻、持续时间、幅值未知 → 需 **JMS** 同时估计状态与模式。
4. **传感器**：本文考虑 **仅角度** 的空间基观测网 → 测量函数强非线性；未检测时含 **负信息**。

因此，单高斯 IMM/EKF 或固定分量的 GMIMM 在 **观测间隙 + 近月机动** 场景下容易丢失 track custody。

**【文字说明】** 地月空间域感知（SDA）的核心任务之一是 **维持 track custody**——即在长时间内持续掌握目标状态及其不确定性。与合作目标不同，非合作 CSO 不会广播辅助信息；监视又常依赖 **被动角度测量**，可观测性弱、非线性强。论文指出：即便目标按弹道飞行，由于 CR3BP 的混沌性，状态 pdf 也会迅速偏离高斯；若再叠加未知机动，传统以单高斯 IMM 为核心的滤波器往往在 **第一次长观测间隙** 后就无法恢复。AGMIMM 的设计目标，正是在这种“动力学混沌 + 观测稀疏 + 模式未知”的三重困难下仍能给出 **可信的非高斯 pdf**。

### 1.2 状态、动力学与 Jacobi 常数

**式 (1)** — 状态定义（会合系 / synodic frame，相对地月质心）：

$$
x_k = [r_k^{\top}, v_k^{\top}]^{\top} = [x_k, y_k, z_k, \dot{x}_k, \dot{y}_k, \dot{z}_k]^{\top}
$$

**式 (2)–(3)** — 归一化单位：$1\,[\text{LU}] = 379632.95\,\text{km}$，$1\,[\text{TU}] = 368232.68\,\text{s}$。

**式 (4)–(7)** — CR3BP 连续动力学（模式相关加速度 $w(t,\tau)$）：

$$
\ddot{x} - 2\dot{y} = w_x(t,\tau) + \frac{\partial U}{\partial x}\bigg|_{r}, \quad \ddot{y} + 2\dot{x} = w_y(t,\tau) + \frac{\partial U}{\partial y}\bigg|_{r}, \quad \ddot{z} = w_z(t,\tau) + \frac{\partial U}{\partial z}\bigg|_{r}
$$

伪势 $U(r)$ 见式 (7)。当 $w=0$ 时为经典 CR3BP。

**式 (8)** — Jacobi 常数（弹道积分量）：

$$
C(r_k, v_k) = 2U(r_k) - \|v_k\|^2
$$

**物理含义**：同一弹道轨迹上 $C$ 恒定；混合分量内 $C$ 的方差大 → 分量内包含不同轨道族 → 线性化传播误差大（见第 5.2 节）。

**【文字说明】** Jacobi 常数 $C$ 是 CR3BP 弹道运动的积分量，可理解为“等效能量”指标：同一轨道上 $C$ 不变，不同轨道则 $C$ 不同。对 GMM 滤波而言，若某个高斯分量内部 $C$ 的散布很大，说明该分量 **同时覆盖多条不同轨道**；用线性化（或统计线性化）把这样一个“胖分量”一次性传播，必然产生较大误差。因此 $C$ 的方差可作为 **是否需要分裂** 的廉价判据——这是 AGMIMM 相对一般 AGM 在地月问题上的专门设计。

### 1.3 模式转移（JMS）

**式 (9)** — 齐次马尔可夫链：

$$
\pi_{ij} = \Pr(\tau_k = j \mid \tau_{k-1} = i)
$$

假设：转移概率已知、时不变、与连续状态无关（算例中可故意违背以测鲁棒性）。

模式 **左连续**：$\tau_k$ 在 $t \in (t_{k-1}^+, t_k^-]$ 内有效。

### 1.4 多传感器仅角度测量与 RFS

每个观测星 $o$ 有 body frame $\mathcal{B}^{(o)}$，相机视轴沿 $\hat{b}_2^{(o)}$。

**式 (11)–(13)** — 检测概率与角度测量：

$$
p_D^{(o)}(x; S^{(o)}) = \mathbf{1}_{S^{(o)}}(r) \times \psi(x, \xi^{(o)})
$$

$$
h^{(o)}(x) = [\arctan(\hat{u}^{(o)\top}\hat{b}_1^{(o)}/\hat{u}^{(o)\top}\hat{b}_2^{(o)}), \arcsin(\hat{u}^{(o)\top}\hat{b}_3^{(o)})]^{\top}
$$

其中 $\hat{u}^{(o)} = (r - r_{\text{obs}}^{(o)})/\|r - r_{\text{obs}}^{(o)}\|$。

**式 (15)** — 多传感器测量为元组 $Z_k = (Z_k^{(1)}, \ldots, Z_k^{(O)})$；每个 $Z_k^{(o)}$ 为 **Bernoulli RFS**：要么空集 $\emptyset$（未检测），要么含单个测量向量。

**动机**：RFS 似然把 **检测概率 + 测量噪声 + 未检测** 统一进 Bayes 更新，使负信息可系统融合（第 4 章详述）。

**【文字说明】** 地月监视中，“没有量测”并不等于“没有信息”。若某观测星的 FoV 覆盖某区域而 **未检测到** 目标，则目标位于该区域内的概率应下降——这称为 **负信息（negative information）**。Bernoulli RFS 把“检测到一点”与“检测为空集”统一写成随机有限集似然；检测概率 $p_D(x)$ 随目标位置、光照、遮挡、月球排除角变化，因而是 **状态相关** 的。AGMIMM 后续对 $p_D$ 的 Taylor 近似与 FoV 边界分裂，都是为了在 GMM 框架下 **正确吸收** 这类信息。

### 1.5 离散 JMS 模型与 Bayes 链

**式 (16)–(17)**：

$$
x_k = f(x_{k-1}) + w_{k-1}(\tau_k), \quad z_k^{(o)} = h^{(o)}(x_k) + \nu_k^{(o)}(\tau_k)
$$

**式 (18)** — 递推链：

$$
\cdots \to p_{k-1}(\tilde{x}_{k-1} \mid Z_{0:k-1}) \to p_{k|k-1}(\tilde{x}_k \mid Z_{0:k-1}) \to p_k(\tilde{x}_k \mid Z_{0:k}) \to \cdots
$$

---

## 第 2 章 非线性 JMS Bayes 滤波（Sect. 3.1）

### 2.1 联合后验与多传感器似然分解

**式 (19)** — Bayes 更新：

$$
p_{k|k}(\tilde{x} \mid Z_{0:k}) = \frac{g_k(Z_k \mid \tilde{x})\, p_{k|k-1}(\tilde{x})}{\int \sum_{\ell=1}^{M} g_k(Z_k \mid \tilde{x}^{(\ell)})\, p(\tilde{x}^{(\ell)})\, d\tilde{x}}
$$

**式 (20)** — 多传感器似然可分解为乘积（条件独立）：

$$
g_k(Z_k \mid \tilde{x}) = \prod_{o=1}^{O} g_k^{(o)}(Z_k^{(o)} \mid \tilde{x})
$$

**式 (21)–(22)** — 更新可写为 **单传感器算子** 的复合：

$$
p_{k|k}^{(\tilde{x}^{(j)})}(\tilde{x}_k \mid Z_{0:k}) = \Psi_k^{(O)} \circ \cdots \circ \Psi_k^{(1)}\, p_{k|k-1}^{(\tilde{x}^{(j)})}(\tilde{x}_k \mid Z_{0:k-1})
$$

$$
[\Psi_k^{(o)} p](\tilde{x}) = \frac{g_k^{(o)}(Z_k^{(o)} \mid \tilde{x})\, p(\tilde{x})}{\int \sum_{\ell} g_k^{(o)}(Z_k^{(o)} \mid \tilde{x}^{(\ell)})\, p(\tilde{x}^{(\ell)})\, d\tilde{x}}
$$

**推导要点**：先对联合似然因式分解，再逐传感器做 Bayes 更新；每步归一化分母含对所有模式的积分，保证概率守恒。

**【文字说明】** 多传感器情况下，若各传感器测量在给定状态下条件独立，则联合似然等于各传感器似然之积。实现上不必构造 $O$ 维联合更新，而可 **按传感器顺序** 依次应用 Bayes 算子 $\Psi_k^{(o)}$——每步只处理一个 $Z_k^{(o)}$，计算量更可控。这与 Vo 等人在 Bernoulli 滤波多传感器融合中的结论一致：序贯更新与联合更新在独立假设下等价。

### 2.2 预测步

**式 (23)** — Chapman–Kolmogorov + 全概率：

$$
p_{k|k-1}(\tilde{x}_k^{(j)} \mid Z_{0:k-1}) = \int \sum_{i=1}^{M} p_{k|k-1}(\tilde{x}_k^{(j)} \mid \tilde{x}_{k-1}^{(i)})\, p_{k-1}(\tilde{x}_{k-1}^{(i)} \mid Z_{0:k-1})\, d\tilde{x}_{k-1}
$$

**复杂度障碍**：精确解需 $O(M^k)$ 项（模式历史指数增长）+ 非线性动力学/测量 → 必须近似（IMM 族）。

**【文字说明】** 若保留全部模式历史，$k$ 步后需 $M^k$ 条假设轨迹，计算与存储均不可行。IMM 的核心近似是：**只保留当前时刻的 $M$ 个模式**，通过交互混合（式 34）把上一时刻各模式的后验“揉”成下一时刻各模式的先验，再分别预测/更新。AGMIMM 在此框架上进一步用 GMM 表示每个模式 pdf，并用 GMRC 控制混合后的分量数。

### 2.3 模式分解与 mode-conditioned 方程

定义 mode-conditioned pdf：

**式 (24)–(25)**：

$$
p_{k|k-1}^{(j)}(x_k) = p_{k|k-1}(x_k \mid \tau_k=j, Z_{0:k-1}), \quad p_k^{(j)}(x_k) = p_k(x_k \mid \tau_k=j, Z_{0:k})
$$

**式 (26)–(27)** — 模式概率：

$$
\mu_{k-1}^{(i)} = \Pr(\tau_{k-1}=i \mid Z_{0:k-1}), \quad \bar{\mu}_{k|k-1}^{(j)} = \sum_{i=1}^{M} \pi_{ij}\, \mu_{k-1}^{(i)}
$$

**式 (28)–(29)** — 单传感器对 mode-conditioned 密度与模式概率的更新算子 $\Psi_k^{(o,j)}$。

**式 (32)–(35)** — 核心 IMM 递推：

$$
p_{k|k-1}^{(j)}(x_k) = \int p_{k|k-1}(x_k \mid x_{k-1}, \tau_k=j)\, p_{k-1}(x_{k-1} \mid \tau_k=j, Z_{0:k-1})\, dx_{k-1}
$$

$$
p_k^{(j)}(x_k) = \Psi_k^{(O,j)} \circ \cdots \circ \Psi_k^{(1,j)}\, p_{k|k-1}^{(j)}(x_k \mid Z_{0:k-1})
$$

$$
\mu_k^{(j)} = \Psi_k^{(O,j)} \circ \cdots \circ \Psi_k^{(1,j)}\, \bar{\mu}_{k|k-1}^{(j)}
$$

**式 (34)** — **混合 mode-conditioned 后验**（IMM 交互步的关键）：

$$
p_{k-1}(x_{k-1} \mid \tau_k=j, Z_{0:k-1}) = \sum_{i=1}^{M} \frac{\pi_{ij}\, \mu_{k-1}^{(i)}}{\bar{\mu}_{k|k-1}^{(j)}}\, p_{k-1}^{(i)}(x_{k-1})
$$

**直觉**：预测到 $t_k$ 时模式为 $j$，但 $t_{k-1}$ 的真实模式可能是 $i$；上式按转移概率 $\pi_{ij}$ 与后验模式权重 $\mu_{k-1}^{(i)}$ 做 **交互混合**。

### 2.4 式 (34) 的推导要点

**目标**：在 $t_k$ 时刻 **假设** 当前模式为 $j$，构造用于预测的模式条件先验 $p(x_{k-1} \mid \tau_k=j, Z_{0:k-1})$。

**Step 1 — 全概率（对 $t_{k-1}$ 的模式求和）**：

$$
p(x_{k-1} \mid \tau_k=j, Z_{0:k-1}) = \sum_{i=1}^{M} p(x_{k-1} \mid \tau_{k-1}=i, \tau_k=j, Z_{0:k-1})\, \Pr(\tau_{k-1}=i \mid \tau_k=j, Z_{0:k-1})
$$

**Step 2 — 标准 IMM 近似**：给定 $\tau_k=j$，$\tau_{k-1}$ 与 $\tau_k$ 在给定 $Z_{0:k-1}$ 下 **近似独立**，故

$$
\Pr(\tau_{k-1}=i \mid \tau_k=j, Z_{0:k-1}) \approx \Pr(\tau_{k-1}=i \mid Z_{0:k-1}) = \mu_{k-1}^{(i)}
$$

**Step 3 — 马尔可夫转移**：$p(x_{k-1} \mid \tau_{k-1}=i, \tau_k=j, \cdot) \approx p_{k-1}^{(i)}(x_{k-1})$，权重由 $\pi_{ij}$ 给出。

**Step 4 — 归一化**：分母 $\bar{\mu}_{k|k-1}^{(j)} = \sum_i \pi_{ij}\mu_{k-1}^{(i)}$ 保证混合 pdf 积分为 1。

> **物理直觉**：IMM 交互步 = “若下一步是机动模式，上一步状态更可能来自各历史模式的加权混合”，而非硬切换。

**【文字说明】** 式 (34) 是 IMM 的 **灵魂**：在预测到 $t_k$ 之前，我们并不知道 $\tau_k$ 的真实值，因此对每个 **候选** 模式 $j$，都要构造一个“若现在处于模式 $j$，则 $t_{k-1}$ 的状态应如何混合而来”的先验。权重 $\pi_{ij}\mu_{k-1}^{(i)}/\bar{\mu}_{k|k-1}^{(j)}$ 同时编码 **马尔可夫转移** 与 **数据驱动的模式后验**。GMIMM/AGMIMM 的区别在于：混合后的 pdf 不再压成单高斯，而是用 GMM 近似，且 AGMIMM 用 GMRC 而非简单删分量。

---

## 第 3 章 IMM 近似与 GMRC 聚类合并（Sect. 3.2）

### 3.1 混合后验的指数增长

设各模式后验为 GM（式 36）：

$$
p_{k-1}^{(i)}(x_{k-1}) = \sum_{\ell=1}^{L_{k-1}^{(i)}} \omega_{k-1}^{(\ell,i)}\, \mathcal{N}(x_{k-1}; m_{k-1}^{(\ell,i)}, P_{k-1}^{(\ell,i)})
$$

代入式 (34) 得 **式 (37)**：混合后验有 $\sum_i L_{k-1}^{(i)}$ 个分量；逐步递推后分量数 **指数增长**。

需近似为 **式 (38)** 的 $L$ 分量 GM（$L \lt L_{\max}$）。

### 3.2 IMM：强制单高斯

**式 (39)** — $L=1$，矩匹配得 $m_{k-1|k-1}^{(j)}, P_{k-1|k-1}^{(j)}$。

**局限**（图 2b/c）：混沌传播后真实 pdf 多峰、厚尾；单高斯 **低估尾部**、在低密度区引入伪质量 → 21 天传播后偏差更大。

**【文字说明】** IMM 把混合后验强行压成 **一个** 高斯，等价于假设“所有模式历史的不确定性可用均值+协方差完全概括”。在近地轨道或许尚可，在 CR3BP 中长时间传播后 pdf 常呈 **多瓣、卷曲** 结构（论文图 2），单高斯无法同时保住峰值与尾部，导致后续预测与更新系统性偏差。

### 3.3 GMIMM：保留 top-r + 合并其余

**式 (40)** — $L=r+1$：保留权重最大的 $r$ 个分量，其余 moment-matching 合并为一个高斯。

**局限**（图 2d/e）：低权重分量可能 **空间上相距很远**，合并后协方差过大；弯曲/多峰分布时合并均值可 **落在真 pdf 支撑之外**。

**【文字说明】** GMIMM 的启发式是“保留重要的几个分量，把其余合成一个”。问题在于：**权重小 ≠ 空间上不重要**——低权重分量可能对应尾部或次要模态，与主峰相距很远。 moment-matching 合并会把它们拉成一个 **巨大协方差** 的高斯，均值甚至落在真 pdf 外面（图 2d）。这是 GMIMM 在长间隙传播中失败的重要原因之一。

### 3.4 AGMIMM：GMRC 聚类合并

**式 (41)** — 用 **Gaussian Mixture Reduction via Clustering (GMRC)** 将式 (37) 约化为 $L$ 分量 GM。

**GMRC 三步**（论文附录 Algorithm 2–4）：

1. **Preprocessing**：Runnalls 贪心合并得 $L$ 个初始聚类中心。
2. **Clustering**：修正 k-means，按 **KL 散度** 将原分量分配到最近中心。
3. **Refinement（可选）**：Newton–Raphson 最小化 NISD；本文为降复杂度 **省略**。

**动机对比**：

| 合并策略 | 依据 | 问题 |
|----------|------|------|
| GMIMM | 权重排序 | 忽略空间结构 |
| **GMRC** | KL 聚类 + 矩匹配 | 保留空间分离的模态，更接近真 pdf（图 2f/g） |

**性质**：$L \to L_{\max}$ 时 AGMIMM $\to$ 精确混合；$L=1$ 时退化为 IMM。

### 3.5 GMRC 算法概要（论文附录）

```text
Algorithm 2  GMRC — Gaussian Mixture Reduction via Clustering

Preprocessing:
  用 Runnalls 贪心 KL 合并，将式 (37) 的 sum_i L^{(i)} 个分量
  初步减至 L 个中心 {omega, m, P}_ell

Clustering (Algorithm 3):
  初始化 assignment 矩阵 C (J x L)
  repeat until convergence:
    对每个原分量 j，按 KL 距离 d( N_j, center_ell ) 分配到最近簇
    对每个簇 ell，用 Algorithm 4 计算 barycenter (omega, m, P)

Output:
  L 分量 GMM 近似式 (41)
```

**Algorithm 4（barycenter）直觉**：簇内所有分量按 KL 加权矩匹配，得到新的 $(\omega, m, P)$，使簇内总质量与一阶、二阶矩一致。

**为何优于 GMIMM 的 top-r 合并？**

- GMIMM：丢弃低权重分量信息 → 合并体协方差膨胀、均值可能 **偏离支撑**。
- GMRC：按 **分布形状相似性（KL）** 聚类 → 保留空间分离的多峰结构，更接近式 (37) 的真 pdf（图 2 Monte Carlo 对比）。

**【文字说明】** GMRC 的核心思想是：合并时应看 **pdf 形状是否相近**，而非仅看权重。KL 散度衡量两个高斯之间的信息差异，用作聚类距离；同一簇内的分量再 moment-match 成一个代表分量。这样，空间上分离的多个模态可以分别保留，避免 GMIMM 式“大团合并”。论文省略 NISD refinement 步以换速度，但即便只用预处理+聚类，已明显优于 GMIMM（图 2f）。

---

## 第 4 章 GMM 预测与状态相关检测更新（Sect. 3.3）

### 4.1 预测步 GM 近似

**式 (42)** — 精确预测（卷积积分，非线性下无闭式）：

$$
p_{k|k-1}^{(j)}(x_k) = \sum_{\ell} \omega_{k-1|k-1}^{(\ell,j)} \int \mathcal{N}(x_k; f(x_{k-1}), Q_{k-1}^{(j)})\, \mathcal{N}(x_{k-1}; m_{k-1|k-1}^{(\ell,j)}, P_{k-1|k-1}^{(\ell,j)})\, dx_{k-1}
$$

**式 (43)–(46)** — mixand 协方差足够小时矩匹配：

$$
p_{k|k-1}^{(j)}(x_k) \approx \sum_{\ell} \omega_{k|k-1}^{(\ell,j)}\, \mathcal{N}(x_k; m_{k|k-1}^{(\ell,j)}, P_{k|k-1}^{(\ell,j)})
$$

$$
\omega_{k|k-1}^{(\ell,j)} = \omega_{k-1|k-1}^{(\ell,j)}
$$

$$
m_{k|k-1}^{(\ell,j)} = \int f(x_{k-1})\, \mathcal{N}(x_{k-1}; m_{k-1|k-1}^{(\ell,j)}, P_{k-1|k-1}^{(\ell,j)})\, dx_{k-1}
$$

$$
P_{k|k-1}^{(\ell,j)} = \int (f(x_{k-1}) - m_{k|k-1}^{(\ell,j)})(\cdot)^{\top}\, \mathcal{N}(\cdot)\, dx_{k-1} + Q_{k-1}^{(j)}
$$

**关键假设**：$P_{k-1|k-1}^{(\ell,j)}$ 足够小，使 $f$ 在分量支撑内近似线性 → **这是 AGM 分裂的根本动机**（第 5 章）。

**【文字说明】** 式 (43)–(46) 本质上是 **非线性传播 + 高斯近似**：对每个分量，在 $m_{k-1|k-1}$ 附近用统计线性化或解流映射近似 $f$，再叠加过程噪声 $Q^{(j)}$。近似成立的前提是：分量协方差足够小，使得 $f$ 在 $1\sigma$ 椭球内近似线性。CR3BP 近月段、大间隙后协方差膨胀都会违反该假设——因此必须在预测前后 **主动分裂**，把大分量拆成多个小分量，使每个子分量满足局部线性条件。

### 4.2 RFS 测量似然与负信息

**式 (48)** — 单传感器 RFS 似然：

$$
g_k^{(o,j)}(Z_k^{(o)} \mid x_k) = \begin{cases} 1 - p_D^{(o)}(x_k; S^{(o)}) & Z_k^{(o)} = \emptyset \\ p_D^{(o)}(x_k; S^{(o)})\, g_k^{(o,j)}(z_k^{(o)} \mid x_k) & Z_k^{(o)} = \{z_k\} \end{cases}
$$

其中 $g_k^{(o,j)}(z \mid x) = \mathcal{N}(z; h^{(o)}(x), R_k^{(j)})$。

**式 (49)** — 更新后 pdf 结构（未归一化）：

$$
[\Psi_k^{(o,j)} p^{(j)}](x) \propto \mathbf{1}_{\emptyset}(Z_k^{(o)}) \sum_{\ell} (1-p_D)\, \omega^{(\ell,j)} \mathcal{N}(x; m^{(\ell,j)}, P^{(\ell,j)}) + (1-\mathbf{1}_{\emptyset}) \sum_{\ell} p_D\, \omega^{(\ell,j)} g(z \mid x)\, \mathcal{N}(x; m^{(\ell,j)}, P^{(\ell,j)})
$$

**直觉**：

- **有测量**：似然 × 先验，类似非线性 Kalman 校正。
- **无测量**：权重乘以 $(1-p_D(m))$ → 在 FoV 内/照明满足处 **降低** 该假设的权重 → **负信息**。

### 4.2.1 RFS 似然 → Bayes 更新的推导

**Step 1**：单分量先验 $p^{(j)}(x) = \omega^{(\ell,j)} \mathcal{N}(x; m, P)$。

**Step 2**：代入式 (48)。若 $Z=\emptyset$，似然 $= 1-p_D(x)$；后验 $\propto (1-p_D(x))\, p(x)$。

**Step 3**：对 $p_D$ 在 $m$ 处常值近似：$(1-p_D(x)) \approx (1-p_D(m))$，得式 (59) 权重缩放。

**Step 4**：若有测量 $Z=\{z\}$，似然 $= p_D(x)\, \mathcal{N}(z; h(x), R)$；在 $m$ 处展开 $p_D$，在 $m$ 附近对 $h$ 统计线性化 → 式 (51)–(53) 与 Kalman 形式同构，但 **权重** 含 $p_D(m)\, q(z)$ 项（检测门控）。

**Step 5**：多分量时对每个 $\ell$ 独立做 Step 3–4，再归一化所有 $\omega_{+}^{(\ell,j)}$。

> **与标准 Kalman 的区别**：(1) 权重随检测概率变化；(2) 空测量也更新权重（负信息）；(3) GMM 需对每个分量重复。

**【文字说明】** 从 Bayes 角度看，空测量 $Z=\emptyset$ 时似然为 $(1-p_D(x))$，后验 $\propto (1-p_D)\, p(x)$。在 $p_D$ 缓变区域，这近似于 **只调整权重、不动均值**（式 59–61）；在 FoV 边界附近 $p_D$ 从 0 跳到 1，必须先把分量 **分裂到边界两侧**，否则常数 $p_D$ 近似会严重失真。有测量时，权重中额外的 $p_D(m)\, q(z)$ 因子体现 **检测门控**：只有在“可被看到”的状态下，测量才强烈拉后验。

### 4.3 检测模型线性化与 GM 更新公式

对 $p_D$ 在均值处 **零阶 Taylor 展开** [12]，得 **式 (50)–(58)**（有测量时）：

$$
\omega_{+}^{(\ell,j)} \propto \omega^{(\ell,j)}\, p_D^{(o)}(m^{(\ell,j)}; S^{(o)})\, q_{+}^{(\ell,j)}(z_k^{(o)})
$$

$$
m_{+}^{(\ell,j)} = m^{(\ell,j)} + K_{+}^{(\ell,j)}(z_k^{(o)} - \hat{z}_{+}^{(\ell,j)}), \quad P_{+}^{(\ell,j)} = \text{Joseph 形式 (53)}
$$

$$
q_{+}^{(\ell,j)}(z) = \mathcal{N}(z; \hat{z}_{+}^{(\ell,j)}, P_{z,+}^{(\ell,j)}), \quad K_{+}^{(\ell,j)} = C_{+}^{(\ell,j)} (P_{z,+}^{(\ell,j)})^{-1}
$$

$\hat{z}_{+}, C_{+}, P_{z,+}$ 由 **式 (55)–(57)** 的统计线性化积分给出。

**无测量**（式 59–61）：$\omega_{+}^{(\ell,j)} \propto \omega^{(\ell,j)}(1-p_D(m^{(\ell,j)}))$，均值协方差不变。

### 4.4 模式概率更新与后验混合

**式 (62)** — 模式概率的单传感器更新（分空/非空测量两种情况）。

**式 (63)–(64)** — 所有传感器更新后的 mode-conditioned 后验与全概率：

$$
p_k^{(j)}(x_k) \approx \sum_{\ell} \omega_k^{(\ell,j)}\, \mathcal{N}(x_k; m_k^{(\ell,j)}, P_k^{(\ell,j)}), \quad p_k(x_k \mid Z_{0:k}) = \sum_{j=1}^{M} \mu_k^{(j)}\, p_k^{(j)}(x_k)
$$

### 4.5 单步更新流程

**【文字说明】** 从实现角度，**预测步** 与 **更新步** 都可能在内部调用分裂算子 $\tilde{G}_c$：预测后按 $c_{\text{SL}}(f)$、$c_{\text{JC}}$ 拆；每个传感器更新前按 FoV 拆、更新后按 $c_{\text{SL}}(h)$ 再拆。模式概率 $\mu^{(j)}$ 与 GMM 参数 **同步** 经各传感器更新——这是 IMM 与 GMM 耦合时容易写错的地方：不要只更新 $p^{(j)}(x)$ 而忘记序贯更新 $\bar{\mu}^{(j)}$。

```text
输入：混合后验 p_{k-1}(x | tau_k=j) 的 GMM，模式先验 mu_bar_{k|k-1}^{(j)}
  ↓
预测：对每个模式 j、每个分量 -> m_{k|k-1}, P_{k|k-1}（+ 分裂，见第5章）
  ↓
for o = 1..O:
    更新 GMM：Psi_k^{(o,j)} -> 式 (50)-(61)
    更新模式概率：Psi_k^{(o,j)} on mu_bar -> 式 (62)
  ↓
输出：p_k^{(j)}(x), mu_k^{(j)}；后验混合 p_k(x)
```

---

## 第 5 章 AGM 分裂准则（Sect. 3.4 + Sect. 4）

### 5.0 分裂算子（式 65–72）

**式 (65)–(66)** — 分裂算子 $\tilde{G}_c$：

$$
\tilde{G}_c[p(x)] = \prod_{i=1}^{I} G_c[\omega^{(i)} \mathcal{N}(x; m^{(i)}, P^{(i)})]
$$

$$
G_c[\omega^{(i)} \mathcal{N}(\cdot)] = \begin{cases} \omega^{(i)} \mathcal{N}(\cdot) & c(\omega^{(i)}, m^{(i)}, P^{(i)}) \le 0 \\ \sum_{\ell=1}^{R} G_c[\omega^{(i,\ell)} \mathcal{N}(x; m^{(i,\ell)}, P^{(i,\ell)})] & \text{otherwise} \end{cases}
$$

对新生分量 **递归** 应用直至无分量满足分裂准则。

**式 (67)–(72)** — 利用 **预生成的一维标准高斯最优分裂库** $\{\tilde{\omega}^{(\ell)}, \tilde{m}^{(\ell)}, \tilde{\sigma}\}$，经缩放、平移、协方差对角化实现多维沿单位方向 $\mathbf{u}$ 的分裂，且 **保持原协方差**（除沿 $\mathbf{u}$ 方向矩调整）。

**【文字说明】** 分裂算子 $\tilde{G}_c$ 是对整个 GMM 逐分量检查准则 $c$：若 $c\le 0$ 则保留，否则拆成 $R$ 个子分量并 **递归** 检查子分量。为加速，Huber 等预先优化 **一维标准高斯** 的分裂参数；运行时通过 Cholesky 白化、沿最优方向 $\mathbf{u}^*$ 缩放平移，把一维解推广到多维。这样分裂在计算上可预测，且与 Progressive Bayes / AGM 系列文献一致。

---

### 5.1 统计线性化误差准则（Sect. 4.1）

**动机**：式 (45)–(46)、(55)–(57) 的线性化/统计线性化只在均值 **小邻域** 内准确；混沌动力学或大协方差分量违反假设。

**式 (73)–(75)** — 统计线性回归：$y = g(x) \approx Gx + b$，最小化 $\text{E}[ee^{\top}] = P_e$。

**式 (76)–(78)** — 分裂准则 [53]：

$$
c_{\text{SL}} = s - s_{\max}, \quad s = \omega^{\Gamma}(1 - e^{-\epsilon})^{1-\Gamma}, \quad \epsilon = \text{tr}(P_e)
$$

- $\Gamma \in [0,1]$：权重 vs 线性化误差的权衡；$s \gt s_{\max}$ 则分裂。
- $g$ 可取 $f$（动力学）或 $h$（测量）。

**【文字说明】** 统计线性化误差准则 $c_{\text{SL}}$ 把 **分量权重** 与 **线性化误差大小** 几何插值成标量 $s$：权重大的分量若线性化误差也大，更“值得”分裂；$\Gamma$ 调节二者相对重要性。$\epsilon=\text{tr}(P_e)$ 来自统计线性回归，可直接用 sigma 点/ cubature 点集估计。该准则在 **预测** 阶段对 $g=f$ 使用，在 **测量更新后** 对 $g=h$ 使用，分别应对动力学与角度测量的非线性。

**分裂方向** — **式 (79)–(82)**：在约束 $\mathbf{u}^{\top} P^{-1} \mathbf{u} = 1$ 下最大化方向导数 Frobenius 范数；白化后化为 **最大特征值特征向量**（Rayleigh-Ritz）。

**推导要点（方向优化）**：

1. 定义非线性度量 $N(u) = u^{\top} H u$，其中 $H = \sum_{i=1}^{n_y} D_i^{\top} D_i$，$D_i$ 为 $g_i$ 的 Hessian。
2. 约束 $u^{\top} P^{-1} u = 1$ 等价于在 **1σ 椭球** 上找最非线性方向（不确定性 + 非线性兼顾）。
3. Cholesky：$P = SS^{\top}$，令 $u = Sy$，约束变为 $\|y\|=1$。
4. Rayleigh-Ritz：$y^*$ 为 $S^{\top} H S$ 的 **最大特征值** 对应特征向量；$u^* = Sy^*$。

**何时用 $g=f$ vs $g=h$？**

| 阶段 | $g$ | 准则 |
|------|-----|------|
| 预测后 | $f$（动力学流） | $c_{\text{SL}}(f)$, $c_{\text{JC}}$ |
| 测量更新后 | $h$（角度模型） | $c_{\text{SL}}(h)$ |

---

### 5.2 Jacobi 常数方差准则（Sect. 4.2）

**动机**：弹道轨迹上 $C$ 恒定；分量内 $C$ 方差大 $\Rightarrow$ 含不同轨道族 $\Rightarrow$ 传播后协方差膨胀（图 3）。

**式 (83)–(84)**：

$$
\sigma_C^2 = \int (C(r,v) - \text{E}[C])^2\, \mathcal{N}(x; m, P)\, dx
$$

$$
c_{\text{JC}} = \min(\sigma_C^2 - \sigma_{C,\max}^2,\; \omega - \omega_{\min})
$$

**实现策略**：Jacobi 准则 **廉价**，用于 **筛选** 待分裂分量；真正分裂方向仍用 5.1 的全非线性度量（动力学 $f$）。

**【文字说明】** 直接计算沿各方向的 Hessian 非线性度量代价高，而 Jacobi 方差 $\sigma_C^2$ 只需在已有回归点上计算 $C$ 的样本方差，几乎不增加开销。策略是 **两阶段**：先用 $c_{\text{JC}}$ 快速圈定“可能违反弹道一致性”的分量，再对它们求 $u^*$ 做分裂。图 3 表明：不同 Jacobi 值的点集传播后协方差膨胀远快于同 Jacobi 邻域——这解释了该准则的有效性。

---

### 5.3 负信息 / FoV 边界分裂（Sect. 4.3）

**动机**：式 (50)–(51) 假设 $p_D$ 在分量支撑内 **近似常数**；大分量横跨 FoV 边界时 $p_D$ 剧烈变化 → 近似失效。

**方法**：几何 **递归分裂** [11,12]，直至分量不再显著 overlap FoV 不连续边界（图 4：分裂后 pdf 在 FoV 边界处呈现清晰结构）。

**【文字说明】** 检测概率 $p_D(x)$ 在 FoV 内外、光照/遮挡边界处 **不连续**。若一个高斯分量横跨边界，把 $p_D$ 在均值处当成常数会系统性错误：FoV 内外的点被赋予相同的检测权重。递归分裂的做法是：只要分量与 $p_D$ 的剧烈变化区重叠，就沿几何法向继续拆，直到每个子分量上 $p_D$ 近似常数。图 4 显示分裂后 pdf 在 FoV 边缘出现 **清晰折线结构**，这正是负信息被正确编码的标志。

---

### 5.4 非线性约束（Sect. 4.4）

**Jacobi 界** — **式 (85)**（已知 $\Delta V$ 容量时）：

$$
C_{\min}(r_0, v_0+\Delta V) \le C(r,v) \le C_{\max}(r_0, v_0-\Delta V)
$$

违反界的分量 **剪枝**；未知 $\Delta V$ 时用保守 $C_{\min}, C_{\max}$（Table 2）。

**月球碰撞约束**：近月分量递归分裂 + 截断月球内部密度（图 5）。

**【文字说明】** 除 Jacobi 界外，还可注入 **硬约束**：目标不应位于月球实体内部。做法与负信息类似——在近月区分裂分量，把落入月球半径内的概率质量 **截断丢弃**（图 5b）。Jacobi 界（图 5a）则反映 $\Delta V$ 容量：超过一定 $C$ 变化意味着机动过大，可在滤波中剔除不合理分量。二者都是 **软/硬先验** 进入 GMM 的示例，体现 AGMIMM 框架的可扩展性。

---

### 5.5 目标模式与离散过程噪声（Sect. 4.5）

- **模式 1（弹道）**：$Q^{(1)}$ 小（$q_w \sim 10^{-3}\,[\text{LU}^2/\text{TU}^3]$）
- **模式 2（机动）**：$Q^{(2)}$ 大（$q_w \sim 1$）→ 用 **过程噪声** 近似未知推力，而非显式 $\Delta v$ 参数

**【文字说明】** 本文不直接在状态中加入推力参数，而用 **JMS 双模式 + 不同 $Q$** 描述机动：弹道模式过程噪声小，机动模式过程噪声大，等价于“允许状态在一步内发生较大随机加速度”。真实机动是脉冲 $\Delta v$，与连续白噪声 **模型不匹配**；论文故意保留这种 mismatch 以检验滤波器鲁棒性——ASNEES 略保守（$\approx 0.4$）部分源于滤波器假设“每步都可能机动”。

**式 (86)–(90)** — 连续噪声 $\Gamma Q^{(j)} \Gamma^{\top}$ 映射到离散 $Q_{k-1}^{(j)}$：

$$
Q_{k-1}^{(j)} = \int_{t_{k-1}}^{t_k} \Phi(t_k,t)\, \Gamma Q^{(j)} \Gamma^{\top}\, \Phi^{\top}(t_k,t)\, dt
$$

本文与式 (6) **耦合数值积分** 式 (89) $\dot{Q} = FQ + QF^{\top} + \Gamma Q^{(j)} \Gamma^{\top}$ 求解。

**Table 2 模式转移**：$\pi_{11}=0.99, \pi_{12}=0.01, \pi_{21}=0.1, \pi_{22}=0.9$。

---

### 5.6 正则化动力学（Sect. 4.6）

**动机**：CR3BP 在地球/月球质心处有奇点；长间隙近月传播需数值稳定。

**式 (91)–(96)** — 虚拟时间 $\gamma$、K–S 变换、8 维增广状态；**式 (97)** 同步传播 $Q^{(j)}$。

Dormand–Prince 5(4) 积分；边界 Jacobi 常数一致性 $\lt 10^{-12}$。

**正则化直觉**：近月时 $r_{\text{Moon}}$ 小 → 式 (91) 使 $dt/d\gamma = r_{\text{Moon}}$ 变小 → 仿真时间在近月区 **放慢** → 积分步长相对物理时间更细 → 避免奇点邻域数值爆炸。

**【文字说明】** 长观测间隙内，滤波器需多次 **纯预测**（无测量）。若数值积分在月球附近发散，GMM 均值与协方差都会失真，后续即使再分裂也无法挽回。K–S 正则化把近月动力学改写到无奇点坐标，并与 $Q^{(j)}$ 的 Lyapunov 方程同步积分（式 97），保证 **传播 + 过程噪声映射** 在同一稳定框架下完成。这是地月 CSO 跟踪实现层面容易被忽视、但对 AGMIMM 成功至关重要的细节。

---

### 5.7 AGMIMM 单步算法流程（Algorithm 1）

**【文字说明】** 下面把前述 IMM 交互、GMRC 合并、正则化预测、多准则分裂、序贯 RFS 更新与约束剪枝 **串成一步** 的可执行顺序。实现时可对照论文 Algorithm 1；$O=1$ 即单传感器 AGMIMM。

```text
Algorithm 1  AGMIMM Filter (one time step k, general O sensors)

1. // 交互混合 (IMM)
   For each mode j = 1..M:
     Construct mixed density (34) from {p_{k-1}^{(i)}, mu_{k-1}^{(i)}}
     Approximate via GMRC -> L-component GMM (41): {omega, m, P}_{k-1|k-1}^{(.,j)}

2. // 预测 + 动力学分裂
   For each mode j, each component ell:
     Propagate (43)-(46) with regularized CR3BP
     Apply splitting G~_c with criteria: c_SL (f), c_JC, constraints

3. // 多传感器序贯更新
   mu_bar^{(j)} <- mu_{k|k-1}^{(j)} from (27)
   For o = 1..O:
     Apply FoV/negative-info splitting on p^{(j)} if needed
     GMM measurement update (50)-(61) on p^{(j)}
     Mode prob update (62) on mu_bar^{(j)}

4. // 测量后分裂 + 剪枝
   Apply c_SL (h), Jacobi/Moon constraints; prune invalid components

5. // 输出
   p_k^{(j)}(x), mu_k^{(j)};  p_k(x) = sum_j mu_k^{(j)} p_k^{(j)}(x)
   State estimate: E[x_k] under p_k
```

---

## 第 6 章 总结与速查

### 6.1 三句话总结

1. AGMIMM = **IMM 模式交互** + **各模式 GMM** + **GMRC 混合合并** + **多准则 AGM 分裂**。  
2. 面向地月 **仅角度多传感器 RFS**，系统融合 **负信息、FoV、Jacobi 与碰撞约束**。  
3. 关键在 **局部线性假设**：通过分裂保持小分量，通过 GMRC 保持混合后验形状，通过 JMS 处理未知机动。

### 6.2 公式速查（核心）

| 编号 | 内容 |
|:----:|------|
| (1)–(8) | 状态、CR3BP、Jacobi 常数 |
| (9) | 模式转移 $\pi_{ij}$ |
| (11)–(17) | 检测概率、角度测量、离散 JMS |
| (19)–(23) | Bayes 更新、预测 |
| (32)–(35) | IMM mode-conditioned 递推 |
| (34),(37)–(41) | 混合后验与 IMM/GMIMM/GMRC |
| (42)–(46) | GMM 预测 |
| (48)–(64) | RFS 更新与后验混合 |
| (65)–(72) | 分裂算子 |
| (73)–(84) | SL / Jacobi 分裂准则 |
| (85)–(97) | 约束、模式噪声、正则化 |
| (103) | 估计误差 |

### 6.3 实现检查清单

1. **交互混合**：式 (34) → GMRC → $L$ 分量/模式。  
2. **预测**：正则化 CR3BP + 式 (43)–(46)；$c_{\text{SL}}(f), c_{\text{JC}}$ 分裂。  
3. **逐传感器更新**：FoV 分裂 → 式 (50)–(62)。  
4. **后处理**：$c_{\text{SL}}(h)$ 分裂；Jacobi/月球剪枝。  
5. **输出**：$\text{E}[x_k]$，$\mu_k^{(j)}$；监控 NEES/ASNEES。

### 6.4 精选参考文献

- [4] LeGrand et al. — 地月 angles-only 跟踪、Jacobi 分裂准则  
- [12] LeGrand & Ferrari — 负信息 GMM 分裂  
- [29] Blom & Bar-Shalom — IMM  
- [30] Laneuville & Bar-Shalom — GMIMM  
- [45,46] GMRC 聚类合并  
- [48,49] Tracker Component Library（GMRC 开源实现）

---

## 附录 A：整理说明

### 相对 `AGMIMM.docx` 与 PDF 的处理

| 方面 | 说明 |
|------|------|
| 结构 | 0–6 章 + 附录；删除原数值算例专章；Algorithm 1 并入第 5.7 节 |
| 公式编号 | 对齐 J. Astronaut. Sci. 2025 论文 (1)–(103) |
| docx 说明 | 各章公式后增加 **【文字说明】** 块；若需与 `AGMIMM.docx` 逐字对齐，请本地运行 [`extract_docx.py`](extract_docx.py) 或 [`extract_docx.ps1`](extract_docx.ps1) 生成 `_docx_extract.txt` 后对照增补 |
| 图表 | 以文字描述 + 图注替代嵌入图（Fig. 1–11 见原 PDF） |

### 主要来源

- `Adaptive Gaussian Mixture Filtering for Multi-sensor.pdf`  
- `AGMIMM.docx`  
- 格式模板：`AAS_2025/111.md`（DPF 笔记）

---

## 附录 B：GitHub 公式渲染说明

| 写法 | 说明 |
|------|------|
| 不用 `\tag{}` | 改用 **式 (7a)** 文字编号 |
| 单行 `$$` | 所有 display 公式一行写完，续行不以 `=` / `-` 开头 |
| $w^{(\ell)\text{-}}$ | 先验上标，避免 `^{(\ell)-}` 解析错误 |
| 后验上标 | 用 `^{\text{+}}`，避免裸 `^+` |
| 期望/数域 | 用 `\text{E}`、`\text{R}^n`，不用 `\mathbb{}` |
| 函数名 | 用 `\text{cholupdate}`、`\text{diag}`，不用 `\operatorname` / `\mathrm` |
| 逆矩阵 | 用 `{...}^{-1}`，少用 `(...)^{-1}` |
| 表格/列表 | 单元格内少写复杂公式；不用 `**$...$**` |
| 比较符 | 行内公式用 `\lt` / `\gt`，不用裸 `<` / `>` |
| 流程图 | 用 ` ```text ` 代码块 |

推送后在 GitHub 网页打开本文件即可查看渲染后的公式。
