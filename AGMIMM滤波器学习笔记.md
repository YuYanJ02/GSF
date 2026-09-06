# 自适应高斯混合交互多模型滤波（AGMIMM）学习笔记

> **论文**：*Adaptive Gaussian Mixture Filtering for Multi-sensor Maneuvering Cislunar Space Object Tracking*  
> **期刊**：The Journal of the Astronautical Sciences (2025) 72:2  
> **DOI**：[10.1007/s40295-024-00478-z](https://doi.org/10.1007/s40295-024-00478-z)  
> **作者**：John L. Iannamorelli, Keith A. LeGrand（Purdue University）  

---

# 3.1 非线性 JMS 贝叶斯滤波器详解

## 动机与背景

在**跳跃马尔可夫系统 (Jump Markov System, JMS)** 中，目标的运动模式（例如“弹道飞行”或“机动推进”）会随时间随机切换。我们不仅要估计目标的连续状态 $`\mathbf{x}_{k}`$（位置、速度），还要推断当前的离散模式 $`\tau_{k}`$。两者是耦合的：不同模式对应不同的动力学方程、过程噪声乃至量测模型。

非线性 JMS 的贝叶斯滤波目标：**递归计算联合状态**与**后验概率密度**：

```math
\widetilde{\mathbf{x}}_{k} = [\mathbf{x}_{k}^{\top},\tau_{k}]^{\top}
```

```math
p_{k}(\widetilde{\mathbf{x}}_{k} \mid Z_{0:k})
```

其中 $`Z_{0:k}`$ 是直到时刻 $`k`$ 的所有多传感器量测集合。

挑战来自三方面：

- **非线性**：动力学 $`\mathbf{f}(\cdot)`$ 和量测 $`\mathbf{h}(\cdot)`$ 均非线性（本文采用圆型限制性三体问题 CR3BP）。
- **模式切换**：模式 $`\tau_{k}`$ 服从马尔可夫链，导致联合后验的混合成分数随指数增长。
- **多传感器与随机检测**：每个传感器可能漏检，需通过随机有限集（RFS）统一处理“未检测到”带来的**负信息**。

第 3.1 节给出了该问题的**精确贝叶斯递推公式**，为后续的子最优近似（IMM、GMIMM、AGMIMM）奠定理论基础。

---

## 核心符号与假设

- 目标连续状态（本文 $`n_{x}=6`$，位置+速度）：

```math
\mathbf{x}_{k} \in \mathbb{R}^{n_{x}}
```

- 离散模式，模式转移概率已知且时不变：

```math
\tau_{k} \in \mathcal{M} = \{1,\dots,M\}
```

```math
\pi_{ij} = \Pr(\tau_{k} = j \mid \tau_{k-1} = i)
```

- 多传感器量测：时刻 $`k`$ 所有传感器的观测构成一个**元组**

```math
Z_{k} \triangleq (Z_{k}^{(1)},\dots,Z_{k}^{(O)})
```

  其中每个 $`Z_{k}^{(o)}`$ 是**伯努利随机有限集**：要么为空 $`\varnothing`$（未检测到目标），要么包含单个量测向量 $`\mathbf{z}_{k}^{(o)}`$。

- 联合量测似然：给定 $`\widetilde{\mathbf{x}}`$ 下，各传感器量测**条件独立**，因此：

```math
g_{k}(Z_{k} \mid \widetilde{\mathbf{x}}) = \prod_{o=1}^{O} g_{k}^{(o)}(Z_{k}^{(o)} \mid \widetilde{\mathbf{x}})
```

---

## 公式 (19) – 贝叶斯后验

```math
p_{k}(\widetilde{\mathbf{x}} \mid Z_{0:k}) = \frac{g_{k}(Z_{k} \mid \widetilde{\mathbf{x}})\, p_{k|k-1}(\widetilde{\mathbf{x}} \mid Z_{0:k-1})}{\int \sum_{\tau'} g_{k}(Z_{k} \mid \mathbf{x}',\tau')\, p_{k|k-1}(\mathbf{x}',\tau' \mid Z_{0:k-1})\, d\mathbf{x}'} \qquad (19)
```

**直觉**：这是标准贝叶斯公式的联合版本。先验 $`p_{k|k-1}`$（预测）与当前量测似然 $`g_{k}`$ 相乘，归一化得到后验。分母中积分和求和对所有连续状态和所有模式进行。

为了显式区分模式，论文用上标 $`(j)`$ 表示条件于 $`\tau_{k}=j`$，例如：

```math
\widetilde{\mathbf{x}}^{(j)} = [\mathbf{x}_{k}^{\top}, j]^{\top}
```

---

## 公式 (20) – 多传感器似然分解

```math
g_{k}(Z_{k} \mid \widetilde{\mathbf{x}}) = \prod_{o=1}^{O} g_{k}^{(o)}(Z_{k}^{(o)} \mid \widetilde{\mathbf{x}}) \qquad (20)
```

**直觉**：给定目标状态和模式，各传感器的观测是独立的（仅通过目标状态耦合）。这个乘积性质使得多传感器更新可以**顺序执行**单传感器更新，极大简化计算。

---

## 公式 (21) – 多传感器贝叶斯更新算子组合

```math
p_{k}(\widetilde{\mathbf{x}}_{k}^{(j)} \mid Z_{0:k}) = \Psi_{k}^{(O)} \circ \cdots \circ \Psi_{k}^{(1)}\, p_{k|k-1}(\widetilde{\mathbf{x}}_{k}^{(j)} \mid Z_{0:k-1}) \qquad (21)
```

其中单传感器贝叶斯更新算子 $`\Psi_{k}^{(o)}`$ 定义为：

```math
[\Psi_{k}^{(o)} p](\widetilde{\mathbf{x}}) = \frac{g_{k}^{(o)}(Z_{k}^{(o)} \mid \widetilde{\mathbf{x}})\, p(\widetilde{\mathbf{x}})}{\int \sum_{i=1}^{M} g_{k}^{(o)}(Z_{k}^{(o)} \mid \widetilde{\mathbf{x}}^{(i)})\, p(\widetilde{\mathbf{x}}^{(i)})\, d\mathbf{x}} \qquad (22)
```

**直觉**：由于量测条件独立，我们可以按任意顺序依次用每个传感器的量测更新当前分布。每个 $`\Psi_{k}^{(o)}`$ 将当前密度 $`p`$（已经融合了前 $`o-1`$ 个传感器的信息）与第 $`o`$ 个传感器的似然相乘并归一化。最终等价于同时使用所有传感器。

---

## 公式 (23) – 预测步（Chapman‑Kolmogorov + 全概率）

```math
p_{k|k-1}(\widetilde{\mathbf{x}}_{k}^{(j)} \mid Z_{0:k-1}) = \int \sum_{i=1}^{M} p_{k|k-1}(\widetilde{\mathbf{x}}_{k}^{(j)} \mid \widetilde{\mathbf{x}}_{k-1}^{(i)})\, p_{k-1}(\widetilde{\mathbf{x}}_{k-1}^{(i)} \mid Z_{0:k-1})\, d\mathbf{x}_{k-1} \qquad (23)
```

**直觉**：这是**预测**的核心。它结合了两个随机过程：

1. **状态转移**（从上时刻模式 $`i`$ 和连续状态 $`\mathbf{x}_{k-1}`$ 转移到当前模式 $`j`$ 和连续状态 $`\mathbf{x}_{k}`$ 的概率密度）：

```math
p_{k|k-1}(\widetilde{\mathbf{x}}_{k}^{(j)} \mid \widetilde{\mathbf{x}}_{k-1}^{(i)})
```

   它由三部分构成：
   - 模式转移概率 $`\pi_{ij}`$
   - 动力学方程：

```math
\mathbf{x}_{k} = \mathbf{f}(\mathbf{x}_{k-1}) + \mathbf{w}_{k-1}(\tau_{k}=j)
```

   - 过程噪声分布（高斯）
2. **上时刻的后验**：

```math
p_{k-1}(\widetilde{\mathbf{x}}_{k-1}^{(i)} \mid Z_{0:k-1})
```

对上一时刻所有可能的连续状态和所有模式求和/积分，得到当前时刻的先验。

**为什么需要这样写？** 因为模式 $`\tau_{k}`$ 不可直接观测，且与连续状态耦合。预测步必须考虑所有可能的模式历史，这正是 JMS 滤波复杂度指数增长的根本原因。

---

## 模式条件分解（第 8 页中间部分）

论文进一步将联合分布分解为**模式条件密度**和**模式概率**：

- 模式条件后验密度：

```math
p_{k}^{(j)}(\mathbf{x}_{k}) \triangleq p_{k}(\mathbf{x}_{k} \mid \tau_{k}=j, Z_{0:k}) \qquad (24)
```

- 模式后验概率：

```math
\mu_{k}^{(j)} \triangleq \Pr(\tau_{k}=j \mid Z_{0:k}) \qquad (26)
```

预测步中的模式先验概率由马尔可夫链给出：

```math
\bar{\mu}_{k|k-1}^{(j)} = \sum_{i=1}^{M} \pi_{ij}\, \mu_{k-1}^{(i)} \qquad (27)
```

**直觉**：将问题拆分为 $`M`$ 个并行的**模式条件滤波器**，每个滤波器负责估计一种模式下的连续状态密度 $`p_{k}^{(j)}(\mathbf{x}_{k})`$，而模式概率 $`\mu_{k}^{(j)}`$ 则根据各模式与量测的匹配程度更新。这种分解是 IMM 类算法的基础。

---

## 单传感器更新算子的模式条件版本（公式 28-32）

对于固定的模式 $`j`$，单传感器更新算子 $`\Psi_{k}^{(o,j)}`$ 作用在模式条件密度 $`p^{(j)}(\mathbf{x})`$ 和模式先验概率 $`\bar{\mu}^{(j)}`$ 上：

```math
[\Psi_{k}^{(o,j)} p^{(j)}](\mathbf{x}) = \frac{g_{k}^{(o,j)}(Z_{k}^{(o)} \mid \mathbf{x})\, p^{(j)}(\mathbf{x})}{\int \sum_{\ell=1}^{M} g_{k}^{(o,\ell)}(Z_{k}^{(o)} \mid \mathbf{x})\, p^{(\ell)}(\mathbf{x})\, d\mathbf{x}} \qquad (28)
```

```math
[\Psi_{k}^{(o,j)} \bar{\mu}^{(j)}] = \frac{p_{k}^{(o,j)}(Z_{k}^{(o)} \mid Z_{0:k-1})\, \bar{\mu}^{(j)}}{\sum_{\ell=1}^{M} p_{k}^{(o,\ell)}(Z_{k}^{(o)} \mid Z_{0:k-1})\, \bar{\mu}^{(\ell)}} \qquad (29)
```

其中：
- 模式 $`j`$ 下的量测似然：

```math
g_{k}^{(o,j)}(Z_{k}^{(o)} \mid \mathbf{x}) = g_{k}^{(o)}(Z_{k}^{(o)} \mid \tau_{k}=j, \mathbf{x})
```

- 模式 $`j`$ 的预测似然（边际）：

```math
p_{k}^{(o,j)}(Z_{k}^{(o)} \mid Z_{0:k-1}) = \int g_{k}^{(o,j)}(Z_{k}^{(o)} \mid \mathbf{x})\, p^{(j)}(\mathbf{x})\, d\mathbf{x}
```

**直觉**：模式条件密度的更新**分母需要对所有模式求和**，这是因为量测可能来自任何模式，模式之间的竞争通过分母中的混合体现。模式概率的更新则是贝叶斯公式直接应用于离散随机变量。

---

## 预测步中的混合（公式 33-34）

将预测步 (23) 用模式条件量写出：

```math
p_{k|k-1}^{(j)}(\mathbf{x}_{k}) = \int p_{k|k-1}(\mathbf{x}_{k} \mid \mathbf{x}_{k-1}, \tau_{k}=j) \; \underbrace{p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1})}_{\text{混合后验}} \, d\mathbf{x}_{k-1} \qquad (33)
```

其中**混合后验**定义为：

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1}) = \sum_{i=1}^{M} \left[ \frac{\pi_{ij} \mu_{k-1}^{(i)}}{\bar{\mu}_{k|k-1}^{(j)}} \, p_{k-1}^{(i)}(\mathbf{x}_{k-1}) \right] \qquad (34)
```


**直觉**：要预测当前模式 $`j`$ 下的状态，我们不知道上一时刻的目标是哪个模式。因此需要将上一时刻所有模式的后验密度 $`p_{k-1}^{(i)}(\mathbf{x}_{k-1})`$ 按照**模式转移概率**和**上时刻模式概率**加权混合，并归一化。这个混合密度就是当前模式 $`j`$ 下进行动力学传播的**初始分布**。

公式 (34) 中的权重：

```math
\frac{\pi_{ij} \mu_{k-1}^{(i)}}{\bar{\mu}_{k|k-1}^{(j)}}
```

是一种**贝叶斯回溯**：给定当前模式 $`j`$，上一时刻模式为 $`i`$ 的概率。这正是 IMM 算法中“混合 (mixing)”步骤的精确表达。

---

## 总结：精确 JMS 贝叶斯滤波的递推结构

1. **输入**：上一时刻的模式条件密度 $`\{p_{k-1}^{(i)}(\mathbf{x})\}`$ 和模式概率 $`\{\mu_{k-1}^{(i)}\}`$。
2. **混合**：对每个当前模式 $`j`$，计算混合后验（公式 34）：

```math
p_{k-1}(\mathbf{x} \mid \tau_{k}=j, Z_{0:k-1})
```

3. **预测**：对每个模式 $`j`$，将混合后验通过动力学方程传播，得到（公式 33）：

```math
p_{k|k-1}^{(j)}(\mathbf{x}_{k})
```
4. **更新**：依次利用每个传感器的量测，通过算子 $`\Psi_{k}^{(o,j)}`$ 更新模式条件密度和模式概率 (公式 28-29, 35)。
5. **输出**：后验模式条件密度 $`p_{k}^{(j)}(\mathbf{x}_{k})`$ 和模式概率 $`\mu_{k}^{(j)}`$。

**为什么需要近似？**  
公式 (34) 中的混合后验是 $`M`$ 个高斯混合（每个 $`p_{k-1}^{(i)}`$ 可能是多个高斯分量的混合）的加权和，其分量数会指数增长。此外，非线性传播和更新无法闭式求解。这正是后续章节（3.2-3.4）引入 IMM、GMIMM 和 AGMIMM 近似的原因。

第 3.1 节的意义在于：它为这些近似提供了一个**精确的理论基准**，并清晰地指明了近似需要牺牲的部分（混合密度的精确表示、非线性处理的线性化等）。

---

# 3.2 Bayesian IMM Approximations 详解

## 动机：指数增长的复杂度

在第 3.1 节的精确 JMS 贝叶斯滤波中，**混合后验**（见公式 34）：

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1})
```

是关键中间量。它把上一时刻所有模式的后验密度按照模式转移概率加权混合，作为当前模式 $`j`$ 下预测步的起始分布。

假设上一时刻每个模式 $`i`$ 的后验密度是一个高斯混合（GM）：

```math
p_{k-1}^{(i)}(\mathbf{x}_{k-1}) = \sum_{\ell=1}^{L_{k-1}^{(i)}} \omega_{k-1}^{(\ell,i)} \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1}^{(\ell,i)}, \mathbf{P}_{k-1}^{(\ell,i)})
```

则根据公式 (34)，混合后验变为：

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1}) = \sum_{i=1}^{M} \sum_{\ell=1}^{L_{k-1}^{(i)}} \frac{\pi_{ij} \mu_{k-1}^{(i)}}{\bar{\mu}_{k|k-1}^{(j)}} \omega_{k-1}^{(\ell,i)} \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1}^{(\ell,i)}, \mathbf{P}_{k-1}^{(\ell,i)})
```

**分量数的爆炸**：
- 初始时每个模式可能只有少量分量。
- 每一次预测-更新循环中，混合步骤将 $`M`$ 个 GM 合并成一个更大的 GM，其分量数为：

```math
\sum_{i=1}^{M} L_{k-1}^{(i)}
```
- 若之后每个分量再经过非线性传播和自适应分裂（后续章节），分量数会指数级增长。  
因此，**必须采用近似**来保持计算可行性。

---

## 三种近似思路

论文对比了三种近似策略，从最简单到最精细：

1. **IMM**：假设每个模式条件密度为单高斯（$`L=1`$），混合后验也用单高斯近似（矩匹配）。
2. **GMIMM**：保留权重最高的 $`r`$ 个分量，其余合并成一个高斯（$`L = r+1`$）。
3. **AGMIMM（本文方法）**：使用**基于聚类的混合约简**（GMRC 算法，附录），用指定的 $`L`$ 个分量最优地逼近混合后验。

下面逐一解析其公式与直觉。

---

## 公式 (36)-(38)：GM 的一般形式

**(36)**

```math
p_{k-1}^{(i)}(\mathbf{x}_{k-1}) = \sum_{\ell=1}^{L_{k-1}^{(i)}} \omega_{k-1}^{(\ell,i)} \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1}^{(\ell,i)}, \mathbf{P}_{k-1}^{(\ell,i)})
```

这是模式 $`i`$ 的后验密度，表示为加权高斯混合。GM 是通用逼近器，可表示任意形状的非高斯分布。

**(37)**

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1}) = \sum_{i=1}^{M} \sum_{\ell=1}^{L_{k-1}^{(i)}} \frac{\pi_{ij} \mu_{k-1}^{(i)}}{\bar{\mu}_{k|k-1}^{(j)}} \omega_{k-1}^{(\ell,i)} \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1}^{(\ell,i)}, \mathbf{P}_{k-1}^{(\ell,i)})
```

混合后验的精确表达式：共有

```math
L_{\text{max}} = \sum_{i} L_{k-1}^{(i)}
```

个分量。

**(38)**

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1}) \approx \sum_{\ell=1}^{L} \omega_{k-1|k-1}^{(\ell,j)} \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1|k-1}^{(\ell,j)}, \mathbf{P}_{k-1|k-1}^{(\ell,j)}), \quad L < L_{\text{max}}
```

近似目标：用更少的 $`L`$ 个分量拟合上面的高分量混合。

**直觉**：我们想要一个“压缩”的 GM，它既能捕捉原始分布的多峰性和尾部，又不至于让分量数失控。下面三种方法就是不同的压缩策略。

---

## IMM 近似：公式 (39)

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1}) \approx \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1|k-1}^{(j)}, \mathbf{P}_{k-1|k-1}^{(j)})
```

其中均值和协方差按矩匹配计算：

```math
\mathbf{m}_{k-1|k-1}^{(j)} = \sum_{i=1}^{M} \frac{\pi_{ij} \mu_{k-1}^{(i)}}{\bar{\mu}_{k|k-1}^{(j)}} \mathbf{m}_{k-1}^{(i)},\quad \mathbf{P}_{k-1|k-1}^{(j)} = \text{类似的加权协方差合并公式}
```

（论文未显式写出，但标准 IMM 中就是用混合的一阶矩和二阶矩。）

**直觉**：IMM 假设在混合后每个模式的条件分布仍然是高斯的。这相当于丢弃了所有非高斯特征（多峰、偏斜、重尾）。在弱非线性、短预测周期下可行，但在混沌动力学和长观测间隙下会严重失真（见图 2b 与图 2a 的对比）。

---

## GMIMM 近似：公式 (40)

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1}) \approx \sum_{\ell=1}^{r+1} \omega_{k-1|k-1}^{(\ell,j)} \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1|k-1}^{(\ell,j)}, \mathbf{P}_{k-1|k-1}^{(\ell,j)})
```

具体方法：
1. 将精确混合 (37) 中的所有分量按权重降序排序。
2. 保留前 $`r`$ 个权重最高的分量不变。
3. 将剩余的 $`L_{\text{max}} - r`$ 个分量通过矩匹配合并成一个高斯分量。
4. 最终得到 $`r+1`$ 个分量。

**直觉**：GMIMM 认识到非高斯性主要来自少数高权重分量，低权重分量对分布形状贡献小，可以打包成一个高斯“背景”。这比单高斯 IMM 更灵活，但仍存在问题：低权重分量可能在空间上分散，强行合并成一个高斯会产生一个**大的协方差**，且均值可能落在低概率区域（见图 2d 所示，合并后的成分偏离了真实分布的支持区）。

---

## AGMIMM 近似：公式 (41) 与 GMRC 聚类

```math
p_{k-1}(\mathbf{x}_{k-1} \mid \tau_{k}=j, Z_{0:k-1}) \approx \sum_{\ell=1}^{L} \omega_{k-1|k-1}^{(\ell,j)} \mathcal{N}(\mathbf{x}_{k-1}; \mathbf{m}_{k-1|k-1}^{(\ell,j)}, \mathbf{P}_{k-1|k-1}^{(\ell,j)})
```
但这里的权重、均值、协方差不是简单的“保留前几个 + 合并其余”，而是通过**聚类算法**（Gaussian Mixture Reduction via Clustering, GMRC，见附录）得到的最优 $`L`$ 分量近似。GMRC 的核心思想：

- **预处理**：先用 Runnalls 的合并算法（基于 KL 散度）获得一个初始 $`L`$ 分量解。
- **聚类**：将原始 $`L_{\text{max}}`$ 个分量分配到 $`L`$ 个簇中，分配依据是 KL 散度或其他距离度量。每个簇对应一个输出分量。
- **优化**（可选）：用 Newton‑Raphson 方法优化簇的权重、均值、协方差，以最小化归一化积分平方距离（NISD）。
- 论文为了速度省略了优化步骤。

**直觉**：GMRC 不是简单截断，而是将原始分布**划分为 $`L`$ 个在概率意义上相近的簇**，每个簇用一个高斯代表。这样能保留多个模式峰和尾部的结构，而不会像 GMIMM 那样把空间分离的低权重分量强行合并成一个发散的成分。

---

# 3.3 状态依赖检测的高斯混合滤波器详解

## 本节目标

在第 3.2 节中，我们通过 GMRC 聚类得到了混合后验的近似 GM（公式 41）。现在，我们要利用这个近似完成 **预测** 和 **多传感器更新**，最终获得当前时刻的后验 GM。核心挑战：

1. **非线性预测**：动力学函数 $`\mathbf{f}(\mathbf{x})`$ 是非线性的，无法直接得到高斯分布的精确传播结果。
2. **状态依赖的检测概率**：

```math
p_{D,k}^{(o)}(\mathbf{x}_{k}; \mathcal{S}^{(o)})
```

与目标位置有关（传感器视场、遮挡、照明等），且不连续（视场边界）。
3. **负信息**：没有检测到目标 ($`Z_{k}^{(o)} = \varnothing`$) 同样提供信息，需要通过贝叶斯更新融入。
4. **多传感器序贯更新**：利用量测条件独立，依次应用单传感器更新算子。

本节给出了这些问题的近似处理方法，核心思想是：**用 GM 表示所有密度，对非线性函数进行局部线性化（如无迹变换或统计线性化），对检测概率做零阶泰勒展开（认为在单个高斯分量内近似常数），并通过自适应分裂（将在 3.4 节详述）保证线性化精度**。

---

## 公式 (42)-(46)：非线性预测的 GM 近似

**精确预测**（公式 42）：
```math
p_{k|k-1}^{(j)}(\mathbf{x}_{k}) = \sum_{\ell=1}^{L_{k-1|k-1}^{(j)}} \int \mathcal{N}\big(\mathbf{x}_{k}; \mathbf{f}(\mathbf{x}_{k-1}), \mathbf{Q}_{k-1}^{(j)}\big) \; \mathcal{N}\big(\mathbf{x}_{k-1}; \mathbf{m}_{k-1|k-1}^{(\ell,j)}, \mathbf{P}_{k-1|k-1}^{(\ell,j)}\big) \, d\mathbf{x}_{k-1}
```

**直觉**：每个高斯分量（均值 $`\mathbf{m}`$，协方差 $`\mathbf{P}`$）经过非线性映射 $`\mathbf{f}(\cdot)`$ 并加上噪声后，输出分布一般不是高斯。但若分量协方差足够小， $`\mathbf{f}(\cdot)`$ 在 $`\mathbf{m}`$ 附近可近似线性，输出仍接近高斯。论文假设这一条件成立（这也是后续自适应分裂的动机：保证分量协方差小到线性化有效）。

**近似预测**（公式 43-46）：
```math
p_{k|k-1}^{(j)}(\mathbf{x}_{k}) \approx \sum_{\ell=1}^{L_{k|k-1}^{(j)}} \omega_{k|k-1}^{(\ell,j)} \mathcal{N}\big(\mathbf{x}_{k}; \mathbf{m}_{k|k-1}^{(\ell,j)}, \mathbf{P}_{k|k-1}^{(\ell,j)}\big)
```

其中：
- **权重不变**（公式 44）：

```math
\omega_{k|k-1}^{(\ell,j)} = \omega_{k-1|k-1}^{(\ell,j)}
```

  预测不改变分量的相对权重（但后续更新会改变）。
- **预测均值**（公式 45）：

```math
\mathbf{m}_{k|k-1}^{(\ell,j)} = \int \mathbf{f}(\mathbf{x}_{k-1}) \, \mathcal{N}\big(\mathbf{x}_{k-1}; \mathbf{m}_{k-1|k-1}^{(\ell,j)}, \mathbf{P}_{k-1|k-1}^{(\ell,j)}\big) \, d\mathbf{x}_{k-1}
```

  这是非线性函数 $`\mathbf{f}`$ 在输入高斯分布下的期望。通常用**无迹变换 (UT)** 或**统计线性化**近似计算（论文第 4.1 节采用统计线性回归，通过 sigma 点求加权平均）。
- **预测协方差**（公式 46）：

```math
\mathbf{P}_{k|k-1}^{(\ell,j)} = \int \big(\mathbf{f}(\mathbf{x}_{k-1}) - \mathbf{m}_{k|k-1}^{(\ell,j)}\big)\big(\cdot\big)^\top \mathcal{N}(\mathbf{x}_{k-1}; \dots) d\mathbf{x}_{k-1} + \mathbf{Q}_{k-1}^{(j)}
```
  第一项是 $`\mathbf{f}(\mathbf{x})`$ 在输入分布下的**协方差**（非线性传播带来的不确定性），第二项是过程噪声。两项相加得到预测协方差。

**关键假设**：每个分量的协方差 $`\mathbf{P}_{k-1|k-1}^{(\ell,j)}`$ 必须足够小，使得 $`\mathbf{f}`$ 在分量支撑集内近似线性。否则，上述高斯近似会引入大误差。这正是 3.4 节自适应分裂要解决的问题：对“太大”的分量进行分裂，减小其协方差。

---

## 公式 (47)-(51)：单传感器更新的基本结构

**先验**（当前模式 $`j`$ 下的预测 GM）：
```math
p^{(j)}(\mathbf{x}) = \sum_{\ell=1}^{L^{(j)}} \omega^{(\ell,j)} \mathcal{N}(\mathbf{x}; \mathbf{m}^{(\ell,j)}, \mathbf{P}^{(\ell,j)}) \qquad (47)
```

**量测似然（RFS 形式）**（公式 48）：
```math
g_{k}^{(o,j)}(Z_{k}^{(o)} \mid \mathbf{x}) = 
\begin{cases}
1 - p_{D,k}^{(o)}(\mathbf{x}; \mathcal{S}^{(o)}), & Z_{k}^{(o)} = \varnothing,\\[0.5em]
p_{D,k}^{(o)}(\mathbf{x}; \mathcal{S}^{(o)}) \; g_{k}^{(o)}(\mathbf{z}_{k}^{(o)} \mid \mathbf{x}), & Z_{k}^{(o)} = \{\mathbf{z}_{k}^{(o)}\}
\end{cases}
```
其中：

```math
g_{k}^{(o)}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z}; \mathbf{h}^{(o)}(\mathbf{x}), \mathbf{R}_{k}^{(j)})
```

**直觉**：量测可能为空（未检测到）或包含一个点。两种情况下，似然都依赖于状态依赖的检测概率 $`p_{D,k}^{(o)}(\mathbf{x})`$。当未检测到时，似然是“看不到目标的概率”，它通过 $`1-p_D`$ 传递**负信息**：若某个区域 $`p_D`$ 高但没有检测到，该区域的后验概率会降低。

**单传感器贝叶斯更新（精确形式）**（公式 49）：
```math
[\Psi_{k}^{(o,j)} p^{(j)}](\mathbf{x}) \propto
\mathbf{1}_{\varnothing}(Z_{k}^{(o)}) \sum_{\ell} (1-p_D(\mathbf{x})) \omega^{(\ell,j)} \mathcal{N}(\mathbf{x};\mathbf{m}^{(\ell)},\mathbf{P}^{(\ell)})
\;+\;
(1-\mathbf{1}_{\varnothing}(Z_{k}^{(o)})) \sum_{\ell} p_D(\mathbf{x}) \omega^{(\ell,j)} g(\mathbf{z}\mid\mathbf{x}) \mathcal{N}(\mathbf{x};\mathbf{m}^{(\ell)},\mathbf{P}^{(\ell)})
```

**直觉**：这是贝叶斯公式直接应用。注意分母被省略（只写到正比于）。分母是归一化常数，它由对所有模式、所有连续状态积分得到，实际计算中会涉及模式概率更新（见公式 62）。

由于 $`p_D(\mathbf{x})`$ 和 $`g(\mathbf{z}\mid\mathbf{x})`$ 都是非线性的，无法保持 GM 形式。论文采用**局部线性化近似**：假设在单个高斯分量的支撑内，$`p_D(\mathbf{x})`$ 近似为常数（等于在均值处的值），且量测函数 $`\mathbf{h}(\mathbf{x})`$ 可线性化（通过 EKF 或 sigma 点）。

---

## 公式 (50)-(61)：非线性量测更新的 GM 近似

当有检测 ($`Z_{k}^{(o)} = \{\mathbf{z}_{k}\}`$) 时，对每个分量进行标准非线性更新（如扩展/无迹卡尔曼更新）：

**更新权重**（公式 50 第一行，似然部分）：
```math
q_{+}^{(\ell,j)} = \frac{\int p_D(\mathbf{x}) g(\mathbf{z}\mid\mathbf{x}) \mathcal{N}(\mathbf{x}; \mathbf{m}^{(\ell)},\mathbf{P}^{(\ell)}) d\mathbf{x}}{\int p_D(\mathbf{x}) \mathcal{N}(\mathbf{x}; \mathbf{m}^{(\ell)},\mathbf{P}^{(\ell)}) d\mathbf{x}}
```
在零阶近似下， $`p_D(\mathbf{x}) \approx p_D(\mathbf{m}^{(\ell)})`$，积分简化为：
```math
q_{+}^{(\ell,j)} \approx \int g(\mathbf{z}\mid\mathbf{x}) \mathcal{N}(\mathbf{x}; \mathbf{m}^{(\ell)},\mathbf{P}^{(\ell)}) d\mathbf{x}
```
这是一个**高斯量测似然的预测分布**，可通过线性化计算。

**更新均值和协方差**（公式 55-57）：
```math
\mathbf{m}_{+}^{(\ell,j)} = \mathbf{m}^{(\ell)} + \mathbf{K}^{(\ell)}(\mathbf{z} - \hat{\mathbf{z}}^{(\ell)})
```
```math
\mathbf{P}_{+}^{(\ell,j)} = \mathbf{P}^{(\ell)} - \mathbf{K}^{(\ell)} \mathbf{S}^{(\ell)} (\mathbf{K}^{(\ell)})^\top
```
其中：
- 量测预测均值：

```math
\hat{\mathbf{z}}^{(\ell)} = \int \mathbf{h}(\mathbf{x}) \mathcal{N}(\mathbf{x}; \mathbf{m}^{(\ell)},\mathbf{P}^{(\ell)}) d\mathbf{x}
```

- 新息协方差：

```math
\mathbf{S}^{(\ell)} = \int (\mathbf{h}(\mathbf{x})-\hat{\mathbf{z}}^{(\ell)})(\cdot)^\top \mathcal{N}(\dots) d\mathbf{x} + \mathbf{R}_{k}^{(j)}
```

- 卡尔曼增益：

```math
\mathbf{K}^{(\ell)} = \int (\mathbf{x}-\mathbf{m}^{(\ell)})(\mathbf{h}(\mathbf{x})-\hat{\mathbf{z}}^{(\ell)})^\top \mathcal{N}(\dots) d\mathbf{x} \; (\mathbf{S}^{(\ell)})^{-1}
```

这些积分同样用 UT 或统计线性化近似。

**当没有检测 ($`Z_{k}^{(o)} = \varnothing`$) 时**，更新公式（50 第二行）更简单：
```math
\omega_{+}^{(\ell,j)} = \omega^{(\ell,j)} \big(1 - p_D(\mathbf{m}^{(\ell)})\big), \quad \mathbf{m}_{+}^{(\ell,j)} = \mathbf{m}^{(\ell)},\quad \mathbf{P}_{+}^{(\ell,j)} = \mathbf{P}^{(\ell)}
```
即**只降低权重**（因为未检测到，且检测概率高意味着该分量应被惩罚），分布形状不变。这正是负信息的典型效果：它不能改变连续状态的分布形状（因为没有量测值），但能调整不同区域的相对可能性。

---

## 公式 (62)：模式概率的更新

单传感器更新不仅改变连续状态 GM，也更新模式概率：
```math
\bar{\mu}_{+}^{(j)} \propto
\begin{cases}
\bar{\mu}^{(j)} \sum_{\ell} \omega^{(\ell,j)} \big(1 - p_D^{(o)}(\mathbf{m}^{(\ell,j)})\big), & Z_{k}^{(o)} = \varnothing,\\[0.5em]
\bar{\mu}^{(j)} \sum_{\ell} \omega^{(\ell,j)} p_D^{(o)}(\mathbf{m}^{(\ell,j)}) \, q_{+}^{(\ell,j)}(\mathbf{z}_{k}^{(o)}), & Z_{k}^{(o)} = \{\mathbf{z}_{k}\}
\end{cases}
```
其中 $`q_{+}^{(\ell,j)}`$ 是量测似然的预测积分（与公式 50 中相同）。归一化因子是所有模式 j 的分子之和。

**直觉**：模式概率的更新反映了各模式对当前量测（或未检测）的**边缘似然**。若某个模式 j 的分量在检测概率高的地方却没有检测到，其边缘似然会降低，导致模式概率下降。这允许滤波器自动识别目标当前最可能的运动模式。

---

## 序贯多传感器更新

公式 (21) 已指出，多传感器更新可写作单传感器更新算子的复合：
```math
p_{k}^{(j)}(\mathbf{x}_{k}) = \Psi_{k}^{(O,j)} \circ \cdots \circ \Psi_{k}^{(1,j)} \; p_{k|k-1}^{(j)}(\mathbf{x}_{k})
```
对应的模式概率也依次更新。最终输出（公式 63）：
```math
p_{k}^{(j)}(\mathbf{x}_{k}) \approx \sum_{\ell=1}^{L_{k}^{(j)}} \omega_{k}^{(\ell,j)} \mathcal{N}(\mathbf{x}_{k}; \mathbf{m}_{k}^{(\ell,j)}, \mathbf{P}_{k}^{(\ell,j)})
```

---

## 公式 (64)：最终状态估计

```math
p_{k}(\mathbf{x}_{k} \mid Z_{0:k}) = \sum_{j=1}^{M} \mu_{k}^{(j)} \, p_{k}^{(j)}(\mathbf{x}_{k})
```
即所有模式条件密度的**加权混合**，权重为后验模式概率。

---

## 本节核心思想的直观总结

1. **非线性预测**：将 GM 的每个分量独立地用高斯近似传播（通过 UT/线性化）。为保证近似精度，要求分量协方差小 → 3.4 节分裂。
2. **状态依赖检测概率**：在单个分量内视作常数（零阶近似），从而将负信息更新简化为权重的缩放。
3. **负信息**：未检测到的传感器不改变状态分布形状，只降低预测检测概率高的分量的权重（包括模式概率）。
4. **序贯更新**：利用条件独立，依次处理每个传感器，避免了联合高维积分。
5. **模式概率更新**：通过比较各模式的边缘似然，自动识别当前运动模式。

这些近似虽然在数学上并非精确，但在合理的设计下（结合自适应分裂）能够有效处理混沌三体问题中的长观测间隙和未知机动。第 3.4 节将进一步介绍如何通过**分裂**来满足线性化所需的“小协方差”假设。

---

# 3.4 高斯混合分裂 (Gaussian Mixture Splitting) 详解

## 为什么需要分裂？

第 3.3 节在处理非线性预测和更新时，多次假设**每个高斯分量的协方差足够小**，从而保证：
- 非线性函数 $`\mathbf{f}(\cdot)`$ 在分量支撑集内近似线性（预测步）。
- 检测概率 $`p_D(\mathbf{x})`$ 在分量支撑集内近似常数（更新步）。
- 量测函数 $`\mathbf{h}(\mathbf{x})`$ 可线性化（更新步）。

然而在实际的混沌三体问题中，由于：
- 长观测间隙导致协方差自然增长；
- 强非线性动力学使初始小协方差迅速“拉长”、“弯曲”；
- 视场边界和遮挡引入不连续性；

分量的协方差可能会变得很大，导致上述线性化假设失效。**分裂**的核心思想：**将一个大协方差、可能非线性的高斯分量，替换为若干个更小协方差的高斯分量的加权和**，这些子分量可以更精确地通过非线性变换，从而提高整体滤波精度。

---

## 分裂算子 $`\tilde{G}_c`$ 的数学定义

公式 (65)-(66) 定义了分裂算子：
```math
\tilde{G}_c[p(\mathbf{x})] = \sum_{i=1}^{I} G_c\big[\omega^{(i)}\mathcal{N}(\mathbf{x};\mathbf{m}^{(i)},\mathbf{P}^{(i)})\big]
```

其中对单个分量的操作 $`G_c`$ 为：
```math
G_c\big[\omega^{(i)}\mathcal{N}(\mathbf{x};\mathbf{m}^{(i)},\mathbf{P}^{(i)})\big] =
\begin{cases}
\omega^{(i)}\mathcal{N}(\mathbf{x};\mathbf{m}^{(i)},\mathbf{P}^{(i)}), & c(\omega^{(i)},\mathbf{m}^{(i)},\mathbf{P}^{(i)}) \le 0,\\[0.5em]
\displaystyle \sum_{\ell=1}^{R} G_c\big[\omega^{(i,\ell)}\mathcal{N}(\mathbf{x};\mathbf{m}^{(i,\ell)},\mathbf{P}^{(i,\ell)})\big], & \text{otherwise}.
\end{cases}
```

- $`c(\cdot)`$ 是**分裂判据**（≤0 表示无需分裂，>0 表示需要分裂）。
- $`R`$ 是每次分裂产生的子分量个数（论文中为固定值，如 2 或 3）。
- 分裂是**递归**的：新产生的子分量若仍然满足分裂条件，会继续分裂，直到所有分量均不满足条件。

**直觉**：这类似于一个树形划分过程，每个不满足精度要求的高斯“叶子”被替换为更细的高斯混合，直到整个分布达到局部线性化的精度门槛。

---

## 分裂参数：如何生成子分量？

直接对任意高斯分布求最优分裂参数非常困难。论文采用一个巧妙的方法：**预计算标准正态分布** $`\mathcal{N}(x;0,1)`$ **的最优分裂**，然后通过**线性变换**将分裂参数推广到任意多元高斯。

**标准正态分裂库** (公式 67)：
```math
q(x) = \mathcal{N}(x;0,1) \approx \tilde{q}(x) = \sum_{\ell=1}^{R} \tilde{\omega}^{(\ell)} \mathcal{N}(x; \tilde{m}^{(\ell)}, \tilde{\sigma}^2)
```
- $`\tilde{\omega}^{(\ell)}`$、$`\tilde{m}^{(\ell)}`$、$`\tilde{\sigma}`$ 是预计算的常数，可通过离线优化得到（例如最小化 KL 散度或积分平方误差）。
- 注意：所有子分量使用**相同的标准差** $`\tilde{\sigma}`$，但均值不同。这种结构使得后续向多元扩展时，可以保持原始协方差的“形状”不变，只沿分裂方向增加分辨率。

**直觉**：标准正态分裂相当于将一维单位方差的高斯分解为几个方差更小的子高斯，它们在数轴上错开排列，总体保持原分布的均值和（近似）方差。

---

## 将一维分裂扩展到多元高斯

对于任意多元高斯分量：

```math
\omega^{(i)}\mathcal{N}(\mathbf{x};\mathbf{m}^{(i)},\mathbf{P}^{(i)})
```

我们希望在**某个方向 $`\mathbf{u}`$**（单位向量）上进行分裂，而保留垂直于 $`\mathbf{u}`$ 的方向不变。分裂后的子分量形式为 (公式 68-72)：

```math
\omega^{(i)}\mathcal{N}(\mathbf{x};\mathbf{m}^{(i)},\mathbf{P}^{(i)}) \approx \sum_{\ell=1}^{R} \omega^{(i,\ell)} \mathcal{N}(\mathbf{x};\mathbf{m}^{(i,\ell)}, \mathbf{P}^{(i,\ell)})
```

具体参数：

1. **子分量权重** (公式 69)：

```math
\omega^{(i,\ell)} = \tilde{\omega}^{(\ell)} \cdot \omega^{(i)}
```

   即原权重按标准正态分裂的权重比例分配到子分量。

2. **子分量均值** (公式 70)：

```math
\mathbf{m}^{(i,\ell)} = \mathbf{m}^{(i)} + \tilde{m}^{(\ell)} \, \sigma_u^{(i)} \, \mathbf{u}
```

   -

```math
\sigma_u^{(i)} = \frac{1}{\sqrt{\mathbf{u}^\top (\mathbf{P}^{(i)})^{-1} \mathbf{u}}}
```

  是原分量在方向 $`\mathbf{u}`$ 上的**标准差**（即沿 $`\mathbf{u}`$ 方向的尺度）。
   - 将标准正态的偏移量 $`\tilde{m}^{(\ell)}`$ 缩放到该尺度，再沿 $`\mathbf{u}`$ 方向移动均值。

3. **子分量协方差** (公式 71)：

```math
\mathbf{P}^{(i,\ell)} = \mathbf{P}^{(i)} - (\sigma_u^{(i)})^2 \left( \sum_{j=1}^{R} (\tilde{m}^{(j)})^2 \tilde{\omega}^{(j)} \right) \mathbf{u}\mathbf{u}^\top
```

   - 这个公式的含义：从原始协方差中**减去**沿 $`\mathbf{u}`$ 方向的一部分方差，因为分裂后，子分量在 $`\mathbf{u}`$ 方向上的方差应该比原分量小（这正是分裂的目的）。被减去的量是 $`\sigma_u^2`$ 乘以一个常数因子：

```math
c_{\text{var}} = \sum_j (\tilde{m}^{(j)})^2 \tilde{\omega}^{(j)}
```

   - 注意：所有子分量使用**相同的** $`\mathbf{P}^{(i,\ell)}`$（与 $`\ell`$ 无关），它们只在均值上沿 $`\mathbf{u}`$ 方向错开。
   - 这样设计可以保证分裂前后总体的**一阶矩和二阶矩不变**（即整体均值和协方差守恒），从而近似保持原始分布的整体特征。

4. **方向标准差** (公式 72)：

```math
\sigma_u^{(i)} = \frac{1}{\sqrt{\mathbf{u}^\top (\mathbf{P}^{(i)})^{-1} \mathbf{u}}}
```
   这是原分量在方向 $`\mathbf{u}`$ 上的标准差（沿该方向的“宽度”）。若 $`\mathbf{P}^{(i)}`$ 是各向同性的， $`\sigma_u`$ 就是 $`\sqrt{\lambda}`$ 其中 $`\lambda`$ 是特征值；一般情况它度量了沿该方向的伸长程度。

**直觉**：整个分裂过程相当于**仅沿着某个最关心的方向将高斯“切”成几片**，每片更窄（协方差在该方向减小），并且彼此错开，从而能够捕捉非线性函数在该方向上的变化细节。

---

## 分裂判据 $`c(\cdot)`$

分裂判据有多种，论文第 4 节详细讨论了三个具体判据，这里先概括其目的：

- **统计线性化误差判据 (Sec. 4.1)**：评估利用线性化 $`\mathbf{y} = \mathbf{G}\mathbf{x}+\mathbf{b}`$ 近似非线性函数 $`\mathbf{g}(\mathbf{x})`$ 带来的误差。当误差超过阈值时分裂。
- **Jacobi 常数判据 (Sec. 4.2)**：利用 CR3BP 的守恒量（Jacobi 常数）的方差作为非线性程度的廉价指标，若方差过大则分裂。
- **负信息判据 (Sec. 4.3)**：当高斯分量跨越传感器视场边界时，检测概率 $`p_D`$ 在分量内部变化剧烈，零阶近似失效，需要分裂以精确处理非检测（负信息）。

这些判据的具体形式将在第 4 节详细讲解，此处我们只需理解：**分裂算子需要与一个具体的判据函数 $`c`$ 耦合**，判据返回 >0 表示分量需要分裂。

---

## 分裂方向的选取：公式 (79)-(82)

论文采用**沿最大非线性方向**分裂的策略。对于函数 $`\mathbf{g}(\mathbf{x})`$（可以是动力学 $`\mathbf{f}`$ 或量测 $`\mathbf{h}`$），定义方向 $`\mathbf{u}`$ 的非线性度量：

```math
\left\| \nabla_{\mathbf{u}} \mathcal{G}(\mathbf{x}) \right\|_F^2 = \mathbf{u}^\top \left( \sum_{i=1}^{n_y} \mathbf{D}_i^\top(\mathbf{x}) \mathbf{D}_i(\mathbf{x}) \right) \mathbf{u}
```
- Jacobian 矩阵：

```math
\mathcal{G}(\mathbf{x}) = \nabla_{\mathbf{x}} \mathbf{g}(\mathbf{x})
```

- 第 $`i`$ 个输出分量的 Hessian 矩阵：

```math
\mathbf{D}_i(\mathbf{x}) = \nabla_{\mathbf{x}} \nabla_{\mathbf{x}} g_i(\mathbf{x})|_{\mathbf{x}}
```
- 这个度量实质上是 Hessian 的 Frobenius 范数在方向 $`\mathbf{u}`$ 上的投影，反映了函数在点 $`\mathbf{x}`$ 处沿 $`\mathbf{u}`$ 方向的**曲率**。

为了找到最优分裂方向，我们求解约束优化问题 (公式 80)：

```math
\mathbf{u}^* = \underset{\mathbf{u}^\top \mathbf{P}^{-1} \mathbf{u} = 1}{\arg\max} \; \mathbf{u}^\top \left( \sum_{i=1}^{n_y} \mathbf{D}_i^\top(\mathbf{x}) \mathbf{D}_i(\mathbf{x}) \right) \mathbf{u}
```

约束（将 $`\mathbf{u}`$ 限制在原高斯分量的 **1-σ 椭球** 内，即 Mahalanobis 距离为 1）：

```math
\mathbf{u}^\top \mathbf{P}^{-1} \mathbf{u} = 1
```

这样可以避免选择概率密度极低的远区方向，确保分裂方向在概率显著的区域。

通过白化变换：

```math
\mathbf{y} = \mathbf{S}^{-1}\mathbf{u}
```

其中：

```math
\mathbf{P} = \mathbf{S}\mathbf{S}^\top
```

（如 Cholesky 分解），约束变为 $`\|\mathbf{y}\| = 1`$，问题转化为标准 Rayleigh 商：

```math
\mathbf{y}^* = \underset{\|\mathbf{y}\|=1}{\arg\max} \; \mathbf{y}^\top \mathbf{S}^\top \left( \sum_{i} \mathbf{D}_i^\top \mathbf{D}_i \right) \mathbf{S} \mathbf{y}
```

其解为矩阵 $`\mathbf{S}^\top \left( \sum_i \mathbf{D}_i^\top \mathbf{D}_i \right) \mathbf{S}`$ 的最大特征值对应的特征向量。然后通过 $`\mathbf{u}^* = \mathbf{S}\mathbf{y}^*`$ 变换回原空间。

**直觉**：我们在概率密度最高的区域（1-σ 椭球）内寻找函数弯曲最厉害的方向，沿该方向分裂可以最大程度减少因线性化带来的误差。

---

## 分裂的整体流程（结合算法 1）

在第 3.4 节末尾给出了 AGMIMM 的单步算法（算法 1），其中分裂发生在多个位置：

1. **预测后分裂**：对 $`p_{k|k-1}^{(j)}(\mathbf{x}_k)`$ 应用 $`\tilde{G}_c`$。
2. **每个传感器更新前分裂**：对 $`p^{(j)}(\mathbf{x})`$ 再次分裂。
3. **最终后验分裂**：对所有模式的后验 $`p_{k}^{(j)}(\mathbf{x}_k)`$ 再次分裂。

这意味着：每次非线性操作（预测或更新）前后都会检查分量是否需要分裂，以保证在整个滤波循环中，所有分量的协方差始终小到足以满足局部线性化假设。
