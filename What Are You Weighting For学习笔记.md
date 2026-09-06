# What Are You Weighting For? 学习笔记

> **论文**：*What are You Weighting For? Improved Weights for Gaussian Mixture Filtering With Application to Cislunar Orbit Determination*  
> **会议**：2024 27th International Conference on Information Fusion (FUSION) · IEEE, 2024  
> **预印本**：[arXiv:2405.11081](https://arxiv.org/abs/2405.11081)  
> **作者**：Dalton Durant, Andrey A. Popov, Renato Zanetti（University of Texas at Austin）  
> **原文 PDF**：[`What Are You Weighting For.pdf`](What%20Are%20You%20Weighting%20For.pdf)  
>
> **阅读说明**：GitHub 官方行内公式语法为 `` $`...`$ ``（反引号包裹，避免 `_` 被 Markdown 吃掉）；块级公式用独立 `` ```math `` 代码块，且**公式须写在同一行**（勿用 `\tag{}`，编号用 `\quad (n)` 代替）。

---

## 一句话主旨

在高斯混合滤波中，每个高斯分量仍按标准 EKF、UKF 等方法更新状态；作者只改进分量证据 $`p_i(y)`$ 的近似方法，从而得到更准确的混合权重 $`w_i^+`$。

---

## 1. 贝叶斯主线

先验高斯混合分布：

```math
p^-(x) \approx \sum_{i=1}^{n} w_i^- p_i^-(x)
```

第 $`i`$ 个分量的测量似然、分量证据和总体证据分别为：

```math
p(y \mid x)
```

```math
p_i(y) = \int p(y \mid x)\, p_i^-(x)\, dx
```

```math
p(y) = \sum_{j=1}^{n} w_j^- p_j(y)
```

分量内部的后验为：

```math
p_i(x \mid y) = \frac{p(y \mid x)\, p_i^-(x)}{p_i(y)}
```

混合权重更新为：

```math
w_i^+ = \frac{w_i^- p_i(y)}{\sum_j w_j^- p_j(y)}
```

因此，$`p_i(x \mid y)`$ 决定第 $`i`$ 个分量内部的状态，$`p_i(y)`$ 决定该分量在整个混合分布中的权重。

---

## 2. 公式 (4) 到公式 (5)

对每个分量使用贝叶斯恒等式：

```math
p(y \mid x)\, p_i^-(x) = p_i(y)\, p_i(x \mid y)
```

因此：

```math
p(x \mid y) \propto \sum_i w_i^- p(y \mid x)\, p_i^-(x) = \sum_i w_i^- p_i(y)\, p_i(x \mid y)
```

归一化后：

```math
p(x \mid y) = \sum_i w_i^+ p_i(x \mid y)
```

---

## 3. 为什么需要两次近似

非线性观测模型为：

```math
y = h(x) + \eta, \qquad \eta \sim \mathcal{N}(0, R)
```

当 $`h(x)`$ 非线性时，通常有两个量没有闭式解：

1. 分量后验 $`p_i(x \mid y)`$；
2. 分量证据 $`p_i(y)`$。

**第一次近似**：利用 EKF、UKF 等方法得到

```math
p_i(x \mid y) \approx \mathcal{N}(x;\, \hat{x}_i,\, \hat{P}_i)
```

**第二次近似**：计算用于更新权重的 $`p_i(y)`$。

虽然贝叶斯公式中存在 $`p_i(y)`$，但它对 $`x`$ 只是归一化常数。EKF、UKF 可以利用条件高斯公式直接得到 $`\hat{x}_i, \hat{P}_i`$，**不必先显式求出精确的** $`p_i(y)`$。

---

## 4. 传统权重与改进权重

传统 EKF 型方法在**先验均值** $`\bar{x}_i`$ 处线性化：

```math
p_i(y) \approx \mathcal{N}\!\left(y;\, h(\bar{x}_i),\, \bar{P}_{yy}^{(i)}\right)
```

```math
\bar{P}_{yy}^{(i)} = \bar{H}_i \bar{P}_i \bar{H}_i^{\mathsf{T}} + R
```

作者认为，融合观测后的 $`\hat{x}_i`$ 通常更接近真实状态，因此在**后验均值**处重新线性化：

```math
p_i(y) \approx \mathcal{N}\!\left(y;\, h(\hat{x}_i),\, \hat{P}_{yy}^{(i)}\right)
```

作者**没有**把精确积分改成

```math
\int p(y \mid x)\, p_i(x \mid y)\, dx
```

精确的 $`p_i(y)`$ 仍由先验定义；后验只用于提供更合适的近似位置和协方差信息。

| 方法 | 线性化位置 | 分量状态更新 |
|------|-----------|-------------|
| GM-EKF | 先验均值 $`\bar{x}_i`$ | 标准 EKF |
| GM-EKF* | 后验均值 $`\hat{x}_i`$ | 标准 EKF（不变） |

---

## 5. UKF/CKF 上的拓展

标准 GM-UKF 从先验高斯生成 sigma 点 $`\bar{\chi}_{i\ell}`$，完成分量状态更新，并从先验 sigma 点近似 $`p_i(y)`$。

改进的 **GM-UKF\*** 先执行完全相同的标准 UKF 更新，得到 $`\hat{x}_i, \hat{P}_i`$，再围绕后验生成第二组 sigma 点 $`\hat{\chi}_{i\ell}`$。

作者将证据积分写成重要性采样形式：

```math
p_i(y) = \mathbb{E}_{x \sim p_i(x \mid y)} \left[ \frac{p_i^-(x)\, p(y \mid x)}{p_i(x \mid y)} \right]
```

于是：

```math
\widehat{p}_i(y) \approx \sum_\ell W_\ell \frac{\mathcal{N}(\hat{\chi}_{i\ell};\, \bar{x}_i,\, \bar{P}_i)\, \mathcal{N}(y;\, h(\hat{\chi}_{i\ell}),\, R)}{\mathcal{N}(\hat{\chi}_{i\ell};\, \hat{x}_i,\, \hat{P}_i)}
```

第二组后验 sigma 点**只用于计算混合权重**，不会再次修改 $`\hat{x}_i, \hat{P}_i`$。

---

## 6. 论文真正证明和验证了什么

- **线性测量模型**：证明先验中心权重与后验中心权重**完全等价**。
- **非线性测量模型**：**没有**证明新方法永远更准确。
- **数值实验**：Avocado 示例和地月空间 NRHO 轨道确定示例表明，新权重通常改善 RMSE、KLD 或滤波一致性。
- 分量较少时，准确度改善通常更明显。
- 分量数量增加后，不同方法的 RMSE 可能逐渐接近，但新权重的一致性仍表现较好。
- 方法依赖一个重要假设：**后验估计比先验估计更接近真实状态**，使后验附近的局部近似更加准确。该假设并非总能成立。

---

## 记忆锚点

```math
\boxed{\text{分量状态更新不变} + \text{以后验为参考重新估计证据} = \text{改进混合权重}}
```

---

## 主动回忆问题

先不要看答案，尝试闭卷写出或讲出答案。

1. $`p(y \mid x)`$、$`p_i(y)`$ 和 $`p(y)`$ 分别表示什么？
2. 为什么公式 (10) 的 $`p_i(y)`$ 也可以称为“分量似然”，但不等于 $`p(y \mid x)`$？
3. 公式 (4) 变成公式 (5) 使用了哪个恒等式？
4. 为什么没有先显式计算 $`p_i(y)`$，也能得到近似后验的均值和协方差？
5. 非线性高斯混合滤波中需要进行哪两次近似？
6. 精确的 $`p_i(y)`$ 积分使用先验还是后验？
7. 作者是否直接用 $`\int p(y \mid x)\, p_i(x \mid y)\, dx`$ 代替原来的证据积分？
8. 传统 GM-EKF 与 GM-EKF* 的主要区别是什么？
9. 为什么在后验均值 $`\hat{x}_i`$ 附近线性化可能更准确？
10. 标准 GM-UKF 和 GM-UKF* 的分量状态估计是否不同？
11. GM-UKF* 为什么要计算“先验密度除以后验密度”的比值？
12. GM-UKF* 中的第二组后验 sigma 点用于什么？
13. 在线性情况下，传统权重和改进权重有什么关系？
14. 论文是否证明了非线性情况下改进权重一定更准确？
15. 在什么情况下，作者方法可能不再有明显优势？

---

## 简短答案

1. $`p(y \mid x)`$ 是状态测量似然；$`p_i(y)`$ 是第 $`i`$ 个分量的证据；$`p(y)`$ 是所有分量加权后的总体证据。
2. 它是观测关于离散分量编号 $`i`$ 的似然，即 $`p(y \mid I=i)`$，但已经把连续状态 $`x`$ 积分掉。
3. $`p(y \mid x)\, p_i^-(x) = p_i(y)\, p_i(x \mid y)`$。
4. $`p_i(y)`$ 对 $`x`$ 只是归一化常数；条件高斯更新可以直接给出后验均值和协方差。
5. 近似分量后验 $`p_i(x \mid y)`$，以及近似分量证据 $`p_i(y)`$。
6. 使用先验：$`p_i(y) = \int p(y \mid x)\, p_i^-(x)\, dx`$。
7. 没有。后验仅作为线性化参考或重要性采样分布。
8. 分量内部的 EKF 更新相同；$`p_i(y)`$ 的近似位置和最终混合权重不同。
9. 后验已经融合当前观测，通常更接近观测支持的高概率区域。
10. 基本相同。论文主要替换混合权重计算方法。
11. 因为积分分布从先验改成了后验，必须用重要性比值进行修正。
12. 只用于近似分量证据和更新混合权重。
13. 两者产生相同的归一化权重。
14. 没有；非线性情况下主要由直觉分析和数值实验支持。
15. 后验估计没有比先验更接近真实状态、后验高斯近似较差，或者分量已经足够多且每个分量非常局部时。

---

## 参考文献

```bibtex
@inproceedings{Durant2024weighting,
  title     = {What are You Weighting For? Improved Weights for Gaussian Mixture Filtering},
  author    = {Durant, Dalton and Popov, Andrey A. and Zanetti, Renato},
  booktitle = {2024 27th International Conference on Information Fusion (FUSION)},
  pages     = {1--8},
  publisher = {IEEE},
  year      = {2024}
}

@article{Durant2024weighting_arxiv,
  title         = {What are You Weighting For? Improved Weights for Gaussian Mixture Filtering With Application to Cislunar Orbit Determination},
  author        = {Durant, Dalton and Popov, Andrey A. and Zanetti, Renato},
  journal       = {arXiv preprint arXiv:2405.11081},
  year          = {2024},
  eprint        = {2405.11081},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SP}
}
```
