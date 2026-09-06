# 基于高斯混合的概率初轨确定学习笔记

> **论文**：*Probabilistic Initial Orbit Determination Using Gaussian Mixture Models*  
> **期刊**：Journal of Guidance, Control, and Dynamics, Vol. 36, No. 5, 2013, pp. 1324–1335  
> **DOI**：[10.2514/1.59844](https://doi.org/10.2514/1.59844)  
> **作者**：Kyle J. DeMars, Moriba K. Jah（Missouri University of Science and Technology / AFRL）  
> **原文 PDF**：[`Probabilistic Initial Orbit Determination Using Gaussian.pdf`](Probabilistic%20Initial%20Orbit%20Determination%20Using%20Gaussian.pdf)  
>
> **阅读说明**：GitHub 官方行内公式语法为 `` $`...`$ ``（反引号包裹，避免 `_` 被 Markdown 吃掉）；块级公式用独立 `` ```math `` 代码块，且**公式须写在同一行**（勿用 `\tag{}`，编号用 `\quad (n)` 代替）。

---

## 一句话主旨

单条短弧光学观测无法唯一确定轨道；作者将物理约束可容许域解释为均匀概率分布，用 GMM 逼近该分布，再通过 GMUKF 传播与量测更新，使多峰轨道假设概率随后续观测逐渐收敛。

---

## 一、论文要解决什么问题？

单条短弧光学观测可以获得：

```math
\boldsymbol{y} = (\alpha,\, \delta,\, \dot{\alpha},\, \dot{\delta})
```

但距离 $`\rho`$ 和距离率 $`\dot{\rho}`$ 未知，因此无法唯一确定六维轨道状态。论文的核心思想是：

> 不过早选择一条轨道，而是保留全部物理可行轨道，用概率分布描述，并通过后续观测逐渐排除错误假设。

整体流程：

```math
\text{短弧观测} \rightarrow (\rho,\dot{\rho})\text{可容许域} \rightarrow \text{GMM初始概率密度} \rightarrow \text{GMUKF传播更新} \rightarrow \text{概率逐渐收敛}
```

---

## 二、构造约束可容许域

已知 $`(\alpha,\delta,\dot{\alpha},\dot{\delta})`$ 后，目标位置和速度只取决于 $`(\rho,\dot{\rho})`$。

通过轨道能量条件筛选地球捕获轨道：

```math
\mathcal{E} = \frac{\|\dot{\boldsymbol{r}}\|^2}{2} - \frac{\mu}{\|\boldsymbol{r}\|} < 0
```

还可以加入先验约束：

```math
a \le a_{\max}, \qquad e \le e_{\max}
```

最终得到 $`(\rho,\dot{\rho})`$ 平面中的约束可容许域 $`\mathcal{A}`$。

---

## 三、用 GMM 逼近可容许域

### 1. 概率解释

没有更多先验信息时，作者认为可容许域内各点等可能：

```math
p(\rho,\dot{\rho}) = \begin{cases} 1/\operatorname{Area}(\mathcal{A}), & (\rho,\dot{\rho}) \in \mathcal{A} \\ 0, & \text{其他} \end{cases}
```

该分布边界不规则，不能用单个高斯准确描述，因此采用：

```math
p(\rho,\dot{\rho}) \approx \sum_{\ell=1}^{L} \alpha_\ell\, \mathcal{N}(\boldsymbol{z};\, \boldsymbol{m}_\ell,\, \boldsymbol{P}_\ell), \qquad \boldsymbol{z} = [\rho,\, \dot{\rho}]^{\mathsf{T}}
```

### 2. 一维均匀分布的 GMM 逼近

先研究区间 $`[a,b]`$ 上的均匀分布。为降低优化维数，作者规定：

- 所有分量等权；
- 均值等间距排列；
- 所有分量具有相同方差。

通过最小化均匀分布与 GMM 之间的 $`L_2`$ 距离，求共同标准差：

```math
\min_\sigma \int [p(x) - q(x)]^2\, dx
```

并预先建立"分量数 $`L`$—最优归一化标准差 $`\tilde{\sigma}`$"数据库。

分量数越多，对均匀分布边界的逼近越准确，但计算量越大。

### 3. 从一维扩展到二维可容许域

**第一步**，对距离率积分，得到距离边缘分布：

```math
p_\rho(\rho) = \int p(\rho,\dot{\rho})\, d\dot{\rho}
```

先均匀放置距离方向高斯分量，再通过约束最小二乘求权重：

```math
\min_{\boldsymbol{\alpha}} \|\boldsymbol{p} - \boldsymbol{H}\boldsymbol{\alpha}\|
```

满足：

```math
\boldsymbol{\alpha} \ge 0, \qquad \boldsymbol{1}^{\mathsf{T}}\boldsymbol{\alpha} = 1
```

**第二步**，对每个距离分量中心 $`m_{\rho,\ell}`$，确定可容许域对应的距离率区间：

```math
a_\ell \le \dot{\rho} \le b_\ell
```

再用一维 GMM 逼近该区间的均匀分布。

**最后**组合距离和距离率分量：

```math
\alpha_{\ell k} = \alpha_{\rho,\ell}\, \alpha_{\dot{\rho},k}
```

```math
\boldsymbol{m}_{\ell k} = \begin{bmatrix} m_{\rho,\ell} \\ m_{\dot{\rho},k} \end{bmatrix}, \qquad \boldsymbol{P}_{\ell k} = \begin{bmatrix} P_{\rho,\ell} & 0 \\ 0 & P_{\dot{\rho},k} \end{bmatrix}
```

由此获得覆盖整个可容许域的二维 GMM，并与角度、角速度观测及其协方差组合成六维初始状态分布。

---

## 四、GMUKF 递推

初始或上一时刻后验为：

```math
p(\boldsymbol{x}_{k-1} \mid Y^{k-1}) = \sum_{\ell=1}^{L} \alpha_{\ell,k-1}^{+}\, \mathcal{N}(\boldsymbol{x}_{k-1};\, \boldsymbol{m}_{\ell,k-1}^{+},\, \boldsymbol{P}_{\ell,k-1}^{+})
```

### 1. 传播

对每个高斯分量构造 sigma 点，分别通过非线性轨道动力学传播，再重构：

```math
\boldsymbol{m}_{\ell,k}^{-} = \sum_i w_i\, \boldsymbol{\mathcal{X}}_{\ell,i,k}
```

```math
\boldsymbol{P}_{\ell,k}^{-} = \sum_i w_i (\boldsymbol{\mathcal{X}}_{\ell,i,k} - \boldsymbol{m}_{\ell,k}^{-})(\cdots)^{\mathsf{T}}
```

没有新观测时，分量权重不变：

```math
\alpha_{\ell,k}^{-} = \alpha_{\ell,k-1}^{+}
```

### 2. 分量内部量测更新

将 sigma 点映射到量测空间，计算：

```math
\hat{\boldsymbol{y}}_{\ell,k}^{-}, \qquad \boldsymbol{P}_{\ell,y}, \qquad \boldsymbol{P}_{\ell,xy}
```

卡尔曼增益：

```math
\boldsymbol{K}_{\ell,k} = \boldsymbol{P}_{\ell,xy}\, \boldsymbol{P}_{\ell,y}^{-1}
```

均值和协方差更新：

```math
\boldsymbol{m}_{\ell,k}^{+} = \boldsymbol{m}_{\ell,k}^{-} + \boldsymbol{K}_{\ell,k}(\boldsymbol{y}_k - \hat{\boldsymbol{y}}_{\ell,k}^{-})
```

```math
\boldsymbol{P}_{\ell,k}^{+} = \boldsymbol{P}_{\ell,k}^{-} - \boldsymbol{K}_{\ell,k}\, \boldsymbol{P}_{\ell,y}\, \boldsymbol{K}_{\ell,k}^{\mathsf{T}}
```

### 3. 分量之间的权重更新

第 $`\ell`$ 个轨道假设对实际观测的预测似然为：

```math
\beta_{\ell,k} = \mathcal{N}(\boldsymbol{y}_k;\, \hat{\boldsymbol{y}}_{\ell,k}^{-},\, \boldsymbol{P}_{\ell,y})
```

贝叶斯分母，即总预测观测概率为：

```math
p(\boldsymbol{y}_k \mid Y^{k-1}) = \sum_j \alpha_{j,k}^{-}\, \beta_{j,k}
```

因此：

```math
\boxed{\alpha_{\ell,k}^{+} = \frac{\alpha_{\ell,k}^{-}\, \beta_{\ell,k}}{\sum_j \alpha_{j,k}^{-}\, \beta_{j,k}}}
```

含义：每个分量内部由 UKF 修正轨道状态；不同分量之间由观测似然重新分配概率。

---

## 五、核心认识

GMUKF 同时完成两件事：

1. **连续状态估计**：更新每个轨道假设的均值和协方差；
2. **离散假设判别**：更新不同候选轨道的概率。

它特别适合短弧定轨，因为初始不确定性通常多峰、非椭球，不能用单高斯可靠表示。

工程应用需要补充过程噪声、高阶摄动力、测量系统误差，以及 GMM 分量的剪枝、合并和自适应分裂。

---

## 六、自测问题

1. 为什么一次角度和角速度观测仍不能唯一确定轨道？
2. 可容许域位于哪两个变量构成的平面中？
3. 为什么作者将约束可容许域解释成均匀分布？
4. 第三部分为什么先求距离边缘分布，而不是直接拟合二维 GMM？
5. 一维均匀分布逼近中，作者对权重、均值和方差作了什么限制？
6. GMM 权重 $`\alpha_\ell`$ 与 UKF sigma 点权重 $`w_i`$ 有什么区别？
7. 为什么动力传播阶段 $`\alpha_{\ell,k}^{-} = \alpha_{\ell,k-1}^{+}`$？
8. $`\beta_{\ell,k}`$ 的物理意义是什么？
9. 贝叶斯更新分母为什么等于 $`\sum_j \alpha_{j,k}^{-}\beta_{j,k}`$？
10. GMUKF 相比单一 UKF 多估计了哪一层信息？
11. 如果所有 $`\beta_{\ell,k}`$ 都很小，可能意味着什么？
12. 上千个 GMM 分量用于工程系统时会带来哪些问题？

---

## 简短答案

| # | 关键词 |
|---|--------|
| 1 | 距离和距离率欠观测 |
| 2 | $`(\rho,\dot{\rho})`$ |
| 3 | 无额外先验时等可能 |
| 4 | 二维问题分解 |
| 5 | 等权、等间距、同方差 |
| 6 | 混合概率与数值积分权重 |
| 7 | 传播不产生新证据 |
| 8 | 分量预测观测似然 |
| 9 | 全概率公式 |
| 10 | 轨道假设概率 |
| 11 | 异常观测或模型失配 |
| 12 | 计算量、下溢、剪枝与合并 |

---

## 参考文献

```bibtex
@article{DeMars2013probabilistic,
  author  = {DeMars, Kyle J. and Jah, Moriba K.},
  title   = {Probabilistic Initial Orbit Determination Using Gaussian Mixture Models},
  journal = {Journal of Guidance, Control, and Dynamics},
  volume  = {36},
  number  = {5},
  pages   = {1324--1335},
  year    = {2013},
  doi     = {10.2514/1.59844}
}
```
