# 地月空间自适应 GMM 不确定性传播学习笔记

> **论文**：*Application of Uncertainty Propagation with Adaptive Gaussian Mixture Models for Cislunar Objects*
>
> **会议**：Advanced Maui Optical and Space Surveillance Technologies Conference（AMOS），2025
>
> **作者**：John A. Gaebler、Juan Gutierrez、Paul Billings、Christopher Craft、Charles J. Wetterer、Jason Baldwin、Micah Dilley、Jill Bruer（KBR / Complex Futures / AFRL）
>
> **原文 PDF**：[Application of Uncertainty Propagation with Adaptive Gaussian Mixture Models for Cislunar Objects.pdf](Application%20of%20Uncertainty%20Propagation%20with%20Adaptive%20Gaussian%20Mixture%20Models%20for%20Cislunar%20Objects.pdf)
>
> **速览**：[一页总结与讨论问题](地月空间自适应GMM不确定性传播_一页总结.md)
>
> **阅读说明**：参照本仓库《概率初轨确定GMM学习笔记》的组织方式。行内公式用 `` $`...`$ ``，块级公式用独立的 `` ```math `` 代码块，每个块的公式写在同一行。下文注明“原文式”的编号对应论文；解释性推导用于帮助理解，不是新的实验结论。

---

## 一句话主旨

用统计线性化误差检测每个高斯分量的非线性失真，沿最敏感方向拆分；再利用混合熵结构与 Runnalls 合并代价压缩冗余分量，使 GMM 在不同地月轨道上自动平衡概率密度精度与计算成本。

---

## 一、论文要解决什么问题？

地月空间受到多体引力作用，目标还可能经历数天至数周的观测间隔。初始高斯误差经过长期传播或月球近接后，会拉伸、弯曲，形成单个椭球无法表达的概率分布。

只保存整体均值和协方差，并不等于准确保存整体概率密度。单高斯 EKF/UKF 预测无法表达分布的弯曲、偏斜或多峰结构；蒙特卡洛可以刻画这些结构，但大量轨道积分成本很高。

论文的核心思想是：

> 不预先固定分量数，而是根据当前不确定性与动力学的耦合程度动态调节 GMM：需要细节时拆分，信息重复时合并。

方法主线：

```math
\text{初始高斯} \rightarrow \text{UT传播与统计线性化} \rightarrow \text{非线性检测及方向拆分} \rightarrow \text{熵驱动合并与子分量保护} \rightarrow \text{下一传播时段}
```

**研究边界**：本文重点是无新观测期间的概率密度预测，完整的“预测—观测更新”滤波验证留待后续工作。它也没有在本文中计算碰撞概率。

---

## 二、用 GMM 表示状态不确定性

设六维状态为 $`\boldsymbol{x}=[\boldsymbol{r}^{\mathsf T},\boldsymbol{v}^{\mathsf T}]^{\mathsf T}`$。采用高斯混合近似（原文式 2）：

```math
\hat p(\boldsymbol{x})=\sum_{i=1}^{L}\alpha_i\,\mathcal N(\boldsymbol{x};\boldsymbol{m}_i,\boldsymbol{P}_i),\qquad \alpha_i\ge0,\qquad\sum_{i=1}^{L}\alpha_i=1.
```

- $`L`$：分量数量，决定概率表示的复杂度。
- $`\alpha_i`$：第 $`i`$ 个分量的概率权重。
- $`\boldsymbol{m}_i,\boldsymbol{P}_i`$：该分量的均值与协方差。

多个较小的局部高斯可以沿非线性分布排列。分量越多，表示能力通常越强，但每个分量都要传播一组 sigma 点，总传播负担约随 $`L\,n_{\mathrm{sigma}}`$ 增长。

**不要混淆两类权重**：GMM 权重 $`\alpha_i`$ 表示不同分量的概率质量；UT sigma 点权重用于近似积分、计算一个分量内部的统计量。

无观测且没有拆分/合并时，确定性动力学传播本身不产生新的贝叶斯证据，分量概率权重保持不变；本文的权重调整来自分量重表示，而不是观测似然更新。

---

## 三、统计线性化：什么时候应该拆分？

### 1. 先传播，再寻找最佳线性解释

对单个分量，设非线性传播映射为：

```math
\boldsymbol{y}=f(\boldsymbol{x}).
```

这里 $`\boldsymbol{y}`$ 是传播后的状态，**不是观测量**。通过 UT 得到输入/输出均值、协方差和交叉协方差，再寻找最佳仿射近似（原文式 3–5）：

```math
\boldsymbol{y}\approx\boldsymbol{F}\boldsymbol{x}+\boldsymbol{b},\qquad \boldsymbol{F}=\boldsymbol{P}_{xy}^{\mathsf T}\boldsymbol{P}_x^{-1},\qquad\boldsymbol{b}=\boldsymbol{m}_y-\boldsymbol{F}\boldsymbol{m}_x.
```

其中 $`\boldsymbol{P}_{xy}=\mathbb E[(\boldsymbol{x}-\boldsymbol{m}_x)(\boldsymbol{y}-\boldsymbol{m}_y)^{\mathsf T}]`$。

这是一种考虑输入概率分布的**统计线性回归**，不要把 $`\boldsymbol F`$ 直接等同于某一点的动力学雅可比。实现时只需传播 sigma 点，不要求显式求高阶动力学导数。

### 2. 线性模型无法解释的部分

定义残差（原文式 4）：

```math
\boldsymbol{e}=f(\boldsymbol{x})-(\boldsymbol{F}\boldsymbol{x}+\boldsymbol{b}).
```

在最佳统计线性化下，残差均值为零，并与中心化输入不相关。输出可理解为“线性可解释部分＋残差”：

```math
\boldsymbol{P}_y=\boldsymbol{F}\boldsymbol{P}_x\boldsymbol{F}^{\mathsf T}+\boldsymbol{P}_e.
```

因此得到误差协方差（原文式 6–7）：

```math
\boldsymbol{P}_e=\boldsymbol{P}_y-\boldsymbol{F}\boldsymbol{P}_x\boldsymbol{F}^{\mathsf T},\qquad\epsilon=\mathrm{tr}(\boldsymbol{P}_e).
```

含义：$`\boldsymbol P_y`$ 是非线性传播后的变化，$`\boldsymbol F\boldsymbol P_x\boldsymbol F^{\mathsf T}`$ 是线性模型解释的变化，差值表示剩余的非线性影响。

- 映射在当前分布覆盖范围内近似线性：$`\boldsymbol P_e`$ 小，无需拆分。
- 非线性影响积累：$`\boldsymbol P_e`$ 增大，单个高斯可能需要细化。

这个判据测量的是**动力学非线性与当前分布尺度的耦合**。同一轨道位置，小协方差可能仍近似线性，大协方差却可能覆盖强曲率区域。

### 3. 两个拆分指标

归一化均方误差指标（原文式 8）：

```math
v_{\mathrm{MSE}}=\sqrt{\frac{\epsilon}{\mathrm{tr}(\boldsymbol{P}_y)}}\in[0,1].
```

它度量线性模型不能解释的方差占总输出方差的相对比例。

指数型指标（原文式 9）：

```math
v_{\mathrm{EXP}}=\alpha^\gamma\left(1-\exp(-\epsilon)\right)^{1-\gamma}\in[0,1],\qquad\gamma\in[0,1].
```

$`\gamma`$ 控制分量权重的重要性，可使拆分偏向高权重分量。本文设置 $`\gamma=0`$，因此：

```math
v_{\mathrm{EXP}}=1-\exp(-\epsilon).
```

两种指标超过各自阈值时触发拆分。三个地月算例统一采用：

```math
v_{\mathrm{MSE,th}}=10^{-3},\qquad v_{\mathrm{EXP,th}}=5\times10^{-2}.
```

**注意**：这两个阈值属于不同定义，不能直接按数值大小比较严格程度。作者观察到 EXP 通常更早、更频繁拆分，但这也是在其算例及参数设定下得到的表现，不是所有问题中的普遍定理。

---

## 四、方向拆分：沿哪里拆，拆成多少个？

### 1. 最大方差方向不一定是最佳方向

协方差最大特征值对应的是当前分布最宽的方向，不一定是未来非线性增长最强的方向。因此，作者把协方差各特征向量作为候选方向，再通过动力学传播评估。

设候选单位方向为 $`\hat{\boldsymbol a}_j`$，协方差平方根满足 $`\boldsymbol P_i=\boldsymbol S_i\boldsymbol S_i^{\mathsf T}`$。方向尺度与扰动为（原文式 11）：

```math
\sigma_{\hat a_j}=\|\boldsymbol S_i^{-1}\hat{\boldsymbol a}_j\|_2^{-1},\qquad\boldsymbol\Delta_j=\tilde h\,\sigma_{\hat a_j}\hat{\boldsymbol a}_j,\qquad\tilde h=\sqrt3.
```

### 2. 用二阶响应评估方向敏感性

对均值和正负扰动分别前向传播，计算（原文式 10）：

```math
\boldsymbol{NL}_{i,j}=\frac{f(\boldsymbol m_i+\boldsymbol\Delta_j)+f(\boldsymbol m_i-\boldsymbol\Delta_j)-2f(\boldsymbol m_i)}{2\tilde h^2}.
```

若沿该方向近似线性，正负扰动响应相互抵消，二阶差分接近零；有明显曲率时，响应增大。选择：

```math
j^*=\arg\max_j\|\boldsymbol{NL}_{i,j}\|_2.
```

算例中，方向检验的前向终点为 $`t_s+3\Delta t`$，$`t_s`$ 是指示拆分的时刻。

### 3. 三分量拆分

采用 Vittaldev 与 Russell 的三分量拆分库，将原分量替换为三个较小高斯，其加权和近似原高斯：

```math
\alpha_i\mathcal N(\boldsymbol x;\boldsymbol m_i,\boldsymbol P_i)\approx\sum_{h=1}^{3}\alpha_h\mathcal N(\boldsymbol x;\boldsymbol m_h,\boldsymbol P_h).
```

新分量采用同方差形式，均值沿选定方向展开。这里是**近似重表示**，不要将有限个不同高斯的混合误认为与原高斯处处严格相等。

拆分不是修正错误动力学，而是让较小的局部高斯更容易保持近似线性，从而提高整体非高斯表示能力。

---

## 五、自适应合并：如何控制分量数量？

### 1. 合并后的矩匹配

对分量 $`i,j`$ 定义归一化权重 $`\lambda_i=\alpha_i/(\alpha_i+\alpha_j)`$、$`\lambda_j=1-\lambda_i`$。合并结果（原文式 13–15）为：

```math
\alpha_{ij}=\alpha_i+\alpha_j,\qquad\boldsymbol m_{ij}=\lambda_i\boldsymbol m_i+\lambda_j\boldsymbol m_j.
```

```math
\boldsymbol P_{ij}=\lambda_i\boldsymbol P_i+\lambda_j\boldsymbol P_j+\lambda_i\lambda_j(\boldsymbol m_i-\boldsymbol m_j)(\boldsymbol m_i-\boldsymbol m_j)^{\mathsf T}.
```

最后一项是**分量均值之间的离散散布**，不能遗漏，否则合并后的总体协方差会偏小。矩匹配保持总权重及总体一、二阶矩，但不能保证保存偏度、峰值、尾部等高阶结构。

### 2. Runnalls 成对合并代价

一般 GMM 之间的 KL 散度没有简单闭式表达，因此使用成对 KL 判别量上界（原文式 12）：

```math
D_{\mathrm{uKL}}(i,j)=\frac12\left[(\alpha_i+\alpha_j)\log|\boldsymbol P_{ij}|-\alpha_i\log|\boldsymbol P_i|-\alpha_j\log|\boldsymbol P_j|\right].
```

代价低的分量对优先合并，直到没有分量对低于当前阈值。阈值越大，允许的合并越激进。这个成对上界是对称的，不应把它直接称为两个高斯之间的精确 KL 散度。

### 3. 用混合熵结构选择阈值

采用高斯混合熵上界（原文式 18）：

```math
H_u=\sum_{i=1}^{L}\alpha_i\left[-\log\alpha_i+\frac12\log\left((2\pi e)^{n_d}|\boldsymbol P_i|\right)\right].
```

试探性增大合并阈值 $`\mathcal D`$，沿降阶过程寻找使上界最小的阈值（原文式 19）：

```math
\mathcal D^*=\arg\min_{\mathcal D}H_u\bigl(\text{按阈值 }\mathcal D\text{ 降阶后的混合分布}\bigr).
```

冗余分量被消除时，熵上界通常下降；过度降阶后，单个分量必须变宽，上界可能重新上升。最小点提供混合结构的降阶参考，**不是全局最优概率密度精度的保证**。

### 4. 保存系数限制合并程度

不直接采用最大降阶程度，而引入（原文式 20）：

```math
\mathcal D_s=(1-\Gamma_s)\mathcal D^*,\qquad\Gamma_s\in[0,1].
```

- $`\Gamma_s=0`$：合并到熵优化给出的阈值。
- $`\Gamma_s=1`$：不合并。
- 本文 $`\Gamma_s=0.9`$：实际阈值为 $`0.1\mathcal D^*`$，保守保留细节。

**原文 Fig. 1 的示例**：高地球轨道传播后有 63 个分量，直接按熵最小点降为 3 个时边缘失真；使用保存系数后剩 8 个，边缘保留更好。该示例用于解释合并设计，不是三类地月轨道之一。

### 5. 防止同一步“刚拆开又合回去”

对本步每个新子混合 $`\xi_n`$ 计算内部成对代价，取最小代价（原文式 22–23）：

```math
\mathcal D_e=\min_n\min_{i<j,\ i,j\in\xi_n}D_{\mathrm{uKL}}(i,j),\qquad\mathcal D_s\leftarrow\min(\mathcal D_s,\mathcal D_e).
```

它主要限制同一次拆分所得子分量互相重新合并，并非禁止新分量与所有其他旧分量合并。论文关注的是同一时间步的排除，不是永久保护。

---

## 六、算例、指标与结果

### 1. 实验设置

- **DRO**：远距离逆行轨道，传播 13 天，动力学较平缓，作为基准。
- **LTO**：月球转移轨道，传播 5 天，依次经历地球主导、地月共同作用和月球主导区域。
- **NRHO**：近直线晕轨道，传播 19 天，月球近接是强非线性来源。

参考轨道由地月圆形限制性三体模型生成，DRO/NRHO 在全历表动力学中差分修正。实际传播使用 DE440、EGM2008 4×4、日月点质量引力及太阳光压，$`C_rA/m=0.0109\ \mathrm{m^2/kg}`$；初始协方差由角度观测最小二乘定轨得到并适当缩放。

三个算例使用相同拆分阈值、$`\gamma=0`$、$`\Gamma_s=0.9`$；拆分/合并评估间隔在 DRO/NRHO 为 30 min，在 LTO 为 6 min。因此“统一参数”主要指结构适应参数，**并非所有数值设置完全相同**。

每例传播 10,000 个粒子作参考。初始粒子云还通过分布质量优化减少随机抽样的空隙与团簇。本文不考虑过程噪声。

### 2. 三个评价指标

**LAM（Likelihood Agreement Measure）**：评价预测密度在参考粒子位置的重合程度。均匀权重粒子下（原文式 24–26）：

```math
\mathrm{LAM}=\frac1K\sum_{k=1}^{K}\hat p(\boldsymbol x_{p,k}).
```

同一算例下较高 LAM 表示预测在粒子位置赋予较高密度，但它不是归一化概率，也不是严格的分布距离；不要把它作为唯一精度指标，或忽略单位和尺度直接跨算例比较。

**NNA（Number Not Associated）**：将粒子映射为地基观测的赤经/赤纬，在量测空间计算与所有 GMM 分量的最小马氏距离；用卡方分布的 0.999 分位门限判断是否关联。未关联粒子比例越小，观测关联覆盖越好，但过宽的 GMM 也可能降低 NNA。

**NISE（Normalized Integral Square Error）**：比较一次合并前后 $`\hat p_A,\hat p_B`$ 的密度变化（原文式 27）：

```math
J_{\mathrm{NISE}}=\frac{\int(\hat p_A-\hat p_B)^2\,d\boldsymbol x}{\int\hat p_A^2\,d\boldsymbol x+\int\hat p_B^2\,d\boldsymbol x}.
```

它衡量**合并操作**的失真，不是传播结果相对蒙特卡洛的总误差，也不是碰撞概率误差。

### 3. 结果解读（原文 Table 4）

- **DRO**：末态为 7（MSE）/9（EXP）个分量；NNA 为 0.18%/0.13%，两者 LAM 接近，最大合并 NISE 均低于 $`2\times10^{-3}`$。
- **LTO**：拆分/合并集中在近月末段。EXP 的 LAM 为 $`7.316\times10^{17}`$，MSE 为 $`4.491\times10^{16}`$；NNA 却分别为 1.38% 和 1.20%。EXP 密度更精细，但 MSE 更宽、关联覆盖更保守。
- **NRHO**：约 63 h 和 300 h 的两次月球近接推动分量数增长。第二次近接附近分布拉伸约 50,000 km。EXP/MSE 的 LAM 为 $`4.972\times10^{16}`$/$`2.754\times10^{15}`$，NNA 为 6.84%/2.22%，最大合并 NISE 约 0.015，两种模型的密度形状精度均明显下降。

**为什么更高 LAM 与更低 NNA 不总是同时出现？** 前者偏向密度集中且位置吻合，后者偏向关联范围覆盖。更宽的概率模型可能“漏得少”，却不是更准确地重建真实密度。

原文 Fig. 8、Fig. 11 还展示了合并阈值增加时 NISE 的敏感性：强非线性区域内，稍微增加合并强度就可能丢失大量信息，而实际自适应阈值避开了较高损失区域。

---

## 七、核心认识与局限

该算法同时管理两件事：

1. **局部表示精度**：统计线性化误差决定哪些分量需要细化。
2. **整体表示复杂度**：熵结构与成对合并代价决定哪些冗余分量可以压缩。

主要贡献是将已有统计线性化、方向拆分、熵近似及 Runnalls 降阶方法组合成可调的在线机制，并在不同地月轨道上检验统一结构参数的可用性。它不是提出全新的 GMM 概念，也不是完整观测更新算法。

作者明确指出三类问题：

- **重建会丢失累计非线性**：拆分或合并后重新生成 sigma 点，不能完整保留旧点携带的高阶结构。
- **迟拆分会抹掉偏斜**：原分布已形成“香蕉形”时，对称拆分会丢失偏度，新分量可能落在不合理位置。
- **同方差拆分有局限**：新 sigma 点可能离原分布太远，经历不同动力学，误差被快速放大。

因此，早拆分、频繁拆分可以改善精度，但会增加成本。NRHO 显示了这一矛盾的边界，不能把“覆盖多数粒子”理解为精确恢复全部密度细节。

与本仓库的概率初轨确定笔记相比：前者着重**初始概率分布如何构造、如何随观测更新**；本篇着重**已有状态概率分布如何长期传播、如何自适应改变分量数**。两者可以在同一系统的不同阶段衔接，但并不是同一算法。

---

## 八、自测问题

1. 为什么均值和协方差传播正确，仍可能不能准确描述真实不确定性？
2. GMM 权重 $`\alpha_i`$ 和 UT sigma 点权重有什么区别？
3. 统计线性化的 $`\boldsymbol F`$ 为什么不必等于动力学雅可比？
4. $`\boldsymbol P_e=\boldsymbol P_y-\boldsymbol F\boldsymbol P_x\boldsymbol F^{\mathsf T}`$ 表示什么？
5. MSE 与 EXP 两个指标各自依赖哪些量？本文 $`\gamma=0`$ 意味着什么？
6. 为什么不能直接沿最大协方差特征向量拆分？
7. 方向检验中正负扰动的二阶差分有什么意义？
8. 合并协方差中的均值差外积项为什么不能省略？
9. $`D_{\mathrm{uKL}}`$ 是精确 KL 散度吗？为什么选择低代价分量对？
10. $`\Gamma_s=0,1,0.9`$ 分别对应什么合并行为？
11. 为什么必须加入同一步新子分量的排除阈值？
12. 为什么 EXP 在 NRHO 中 LAM 更高，但 NNA 反而更差？

---

## 简短答案

1. 一、二阶矩不能唯一确定概率分布，无法充分表达弯曲、偏斜和多峰。
2. 前者是分量概率质量，后者是分量内部近似积分的数值权重。
3. 它通过概率加权的最佳仿射回归得到，反映整个分量覆盖范围。
4. 非线性输出协方差中不能被最佳线性映射解释的部分。
5. MSE 使用相对方差比例；EXP 使用指数型误差与可选权重项；本文不按分量权重偏置拆分。
6. 当前最宽方向不一定是未来非线性响应最强方向。
7. 估计沿候选方向的非线性曲率响应；线性映射下该差分为零。
8. 它补充两个分量均值之间的散布，保证总体二阶矩。
9. 是成对合并的 KL 判别量上界；低代价通常意味着较小降阶损失。
10. 分别为按熵优化阈值合并、不合并、采用优化阈值的 10%。
11. 避免合并操作立即抵消因非线性触发的拆分。
12. EXP 密度更集中精细，MSE 更宽更保守；密度重合与关联覆盖是不同评价维度。

---

## 九、讨论问题与可能改进

以下是阅读后提出的研究问题，**不是论文已经验证的结论**：

1. **尺度一致性**：位置与速度分量如何归一化？尤其 EXP 中 $`\exp(-\epsilon)`$ 如何保证尺度解释一致？改变单位或坐标会不会改变触发时机与阈值通用性？
2. **拆分前瞻性**：能否在近月事件之前预判误差增长，结合偏度、多方向或异方差拆分，避免分布已明显弯曲后才重新表示？
3. **自适应保护**：熵上界最小不等于密度误差最小，能否用局部非线性或合并失真预算自动调整 $`\Gamma_s`$，而非固定 0.9？
4. **工程验证**：加入观测更新、过程噪声、异常观测及低权重分量管理后表现如何？需要报告整体耗时、最大分量数、参数敏感性以及多初值统计结果。

---

## 参考文献

主论文：

```bibtex
@inproceedings{Gaebler2025adaptiveCislunar,
  author    = {Gaebler, John A. and Gutierrez, Juan and Billings, Paul and Craft, Christopher and Wetterer, Charles J. and Baldwin, Jason and Dilley, Micah and Bruer, Jill},
  title     = {Application of Uncertainty Propagation with Adaptive Gaussian Mixture Models for Cislunar Objects},
  booktitle = {Advanced Maui Optical and Space Surveillance Technologies Conference},
  year      = {2025}
}
```

方法来源（编号沿用主论文，完整著录见原 PDF）：

- **[5]** M. F. Huber，*Adaptive Gaussian Mixture Filter Based on Statistical Linearization*，2011：统计线性化自适应方法。
- **[13]** V. Vittaldev、R. P. Russell，*Multidirectional Gaussian Mixture Models for Nonlinear Uncertainty Propagation*，2016：方向敏感性与拆分库。
- **[33]** A. R. Runnalls，*Kullback-Leibler Approach to Gaussian Mixture Reduction*，IEEE Transactions on Aerospace and Electronic Systems，43(3)，989–999，2007。
- **[39]** M. F. Huber、T. Bailey、H. Durrant-Whyte、U. D. Hanebeck，*On entropy approximation for Gaussian mixture random vector*，IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems，2008。
