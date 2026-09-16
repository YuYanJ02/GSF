# 熵驱动 AEGIS 非线性不确定性传播学习笔记

> **论文**：*Entropy-Based Approach for Uncertainty Propagation of Nonlinear Dynamical Systems*  
> **期刊**：Journal of Guidance, Control, and Dynamics, Vol. 36, No. 4, 2013, pp. 1047–1057  
> **DOI**：[10.2514/1.58987](https://doi.org/10.2514/1.58987)  
> **作者**：Kyle J. DeMars、Robert H. Bishop、Moriba K. Jah  
> **方法名称**：Adaptive Entropy-Based Gaussian-Mixture Information Synthesis（AEGIS）  
>
> **阅读说明**：本文参照本仓库《概率初轨确定GMM学习笔记》的组织方式。行内公式使用 `` $`...`$ ``，块级公式使用独立的 `` ```math `` 代码块，公式编号沿用原论文。文中标注为“解释性”的内容用于帮助理解，不属于论文新增实验结果。

---

## 一句话主旨

AEGIS 为每个高斯分量同时建立一个线性熵基准和一个非线性 sigma 点预测；当二者的熵差超过阈值时，沿协方差主方向把该分量替换成多个更小的高斯，从而使 GMM 的分量数随非线性强弱在线自适应增长。

---

## 一、论文要解决什么问题？

设状态服从非线性动力学：

```math
\dot{\boldsymbol{x}}(t)=\boldsymbol{f}(\boldsymbol{x}(t),t),\qquad \boldsymbol{x}(t_0)=\boldsymbol{x}_0. \quad (1)
```

即使初始状态是高斯分布，经过长时间非线性传播后，真实概率密度也可能出现：

- 弯曲和“香蕉形”结构；
- 偏斜、厚尾；
- 多峰或狭长流形；
- 位置与速度之间复杂的非线性相关。

单高斯 EKF/UKF 只能保存均值和协方差，无法描述这些形状。固定分量数的 GMM 虽然更灵活，但仍存在两种矛盾：

1. 初始分量太少，进入强非线性区域后表达能力不足；
2. 初始分量过多，在弱非线性阶段浪费大量轨道积分。

AEGIS 的核心问题因此是：

> 能否在传播过程中自动发现某个高斯分量何时开始明显非高斯，并只在此时增加局部分辨率？

整体流程为：

```math
\text{初始GMM}\rightarrow\text{逐分量sigma点传播}\rightarrow\text{熵差检测非线性}\rightarrow\text{超阈值分量拆分}\rightarrow\text{重新生成sigma点并继续传播}.
```

---

## 二、GMM 与不确定性传播问题

### 1. 高斯混合表示

状态概率密度用 GMM 近似（原文式 4）：

```math
p(\boldsymbol{x})=\sum_{i=1}^{L}\alpha_i\,\mathcal N(\boldsymbol{x};\boldsymbol{m}_i,\boldsymbol{P}_i),\qquad \alpha_i\ge0,\qquad\sum_{i=1}^{L}\alpha_i=1. \quad (4)
```

其中：

- $`L`$ 是当前分量数；
- $`\alpha_i`$ 是第 $`i`$ 个分量的概率质量；
- $`\boldsymbol{m}_i`$ 和 $`\boldsymbol{P}_i`$ 是局部均值与协方差。

多个较小的局部高斯可以沿弯曲概率流形排列，因此比单个大高斯更容易保持非线性传播后的分布形状。

### 2. 标准 GMM 传播的局限

标准 GMEKF/GMUKF 通常对每个分量独立传播：

```math
\dot{\boldsymbol{m}}_\ell=\boldsymbol{f}(\boldsymbol{m}_\ell,t),\qquad \dot{\boldsymbol{P}}_\ell=\boldsymbol{F}_\ell\boldsymbol{P}_\ell+\boldsymbol{P}_\ell\boldsymbol{F}_\ell^{\mathsf T},\qquad \dot\alpha_\ell=0.
```

只要没有观测更新或分量重表示，确定性传播不会产生新的贝叶斯证据，所以混合权重保持不变。

但固定 $`L`$ 意味着某个分量即使已经覆盖强非线性区域，也不会自动细化。AEGIS 增加了“检测—拆分”机制，使 $`L`$ 可以在线变化。

---

## 三、用微分熵描述高斯的不确定性体积

### 1. 微分熵

任意连续概率密度 $`p(\boldsymbol{x})`$ 的微分熵定义为：

```math
H(\boldsymbol{x})=-\int p(\boldsymbol{x})\log p(\boldsymbol{x})\,d\boldsymbol{x}=\mathbb E[-\log p(\boldsymbol{x})].
```

对于 $`n`$ 维高斯分布 $`\mathcal N(\boldsymbol{m},\boldsymbol{P})`$，原文式（5）为：

```math
H(\boldsymbol{x})=\frac12\log\left|(2\pi e)\boldsymbol{P}\right|=\frac12\log\left[(2\pi e)^n|\boldsymbol{P}|\right]. \quad (5)
```

熵只依赖协方差行列式，不依赖均值。几何上，$`|\boldsymbol{P}|^{1/2}`$ 与协方差椭球体积成正比，所以这里的熵可理解为高斯不确定性“体积”的对数尺度。

### 2. Rényi 熵

论文也给出 $`\kappa`$ 阶 Rényi 熵（原文式 6–8）。对高斯分布，它同样可以写成协方差行列式的函数。作者说明，后续非线性检测对微分熵和 Rényi 熵都适用；正文主要使用微分熵。

### 3. 重要限制

微分熵主要反映体积，而不是完整形状。两个分布可能具有相同的协方差行列式，却具有不同的方向、弯曲或偏斜。因此熵差是计算便宜的非线性触发指标，而不是对非高斯形状的完备度量。

此外，微分熵会随坐标尺度和单位改变。工程实现中应固定坐标定义，最好先无量纲化，不能直接比较在不同单位体系下计算的阈值。

---

## 四、AEGIS 机制一：传播中检测非线性

### 1. 熵导数的一般形式

由矩阵行列式求导可得（原文式 11）：

```math
\dot H(\boldsymbol{x})=\frac12\operatorname{tr}\!\left(\boldsymbol{P}^{-1}\dot{\boldsymbol{P}}\right). \quad (11)
```

对无过程噪声的线性化系统，协方差满足（原文式 12）：

```math
\dot{\boldsymbol{P}}(t)=\boldsymbol{F}(\boldsymbol{m}(t),t)\boldsymbol{P}(t)+\boldsymbol{P}(t)\boldsymbol{F}^{\mathsf T}(\boldsymbol{m}(t),t). \quad (12)
```

其中动力学雅可比为：

```math
\boldsymbol{F}(\boldsymbol{m},t)=\left.\frac{\partial\boldsymbol{f}(\boldsymbol{x},t)}{\partial\boldsymbol{x}}\right|_{\boldsymbol{x}=\boldsymbol{m}}.
```

将式（12）代入式（11），利用迹的循环性质，可得原文最关键的式（13）：

```math
\boxed{\dot H_{\mathrm{lin}}(t)=\operatorname{tr}[\boldsymbol{F}(\boldsymbol{m}(t),t)]}. \quad (13)
```

这意味着：无过程噪声时，不需要完整运行一个 EKF 协方差预测器，只需沿分量均值轨迹积分一个标量方程：

```math
H_{\mathrm{lin}}(t)=H(t_0)+\int_{t_0}^{t}\operatorname{tr}[\boldsymbol{F}(\boldsymbol{m}(\tau),\tau)]\,d\tau.
```

### 2. 非线性熵预测

同一个高斯分量还使用 sigma 点经过完整非线性动力学传播。传播后重构均值和协方差：

```math
\boldsymbol{m}_{\mathrm{nl}}=\sum_iw_i\boldsymbol{X}_i,qquad \boldsymbol{P}_{\mathrm{nl}}=\sum_iw_i(\boldsymbol{X}_i-\boldsymbol{m}_{\mathrm{nl}})(\boldsymbol{X}_i-\boldsymbol{m}_{\mathrm{nl}})^{\mathsf T}.
```

再用式（5）计算非线性预测器对应的高斯近似熵：

```math
H_{\mathrm{nl}}(t)=\frac12\log\left[(2\pi e)^n|\boldsymbol{P}_{\mathrm{nl}}(t)|\right].
```

这里的 $`H_{\mathrm{nl}}`$ 是“与 sigma 点预测均值和协方差相同的高斯”的熵，不是传播后真实非高斯密度的精确熵。

### 3. 熵差触发器

定义：

```math
\Delta H(t)=|H_{\mathrm{nl}}(t)-H_{\mathrm{lin}}(t)|.
```

当：

```math
\Delta H(t)>\Delta H_{\mathrm{th}}
```

就认为该分量受到可辨识的非线性影响：

1. 停止当前传播；
2. 记录触发时刻 $`t_s`$；
3. 拆分超阈值分量；
4. 为新分量生成 sigma 点；
5. 从 $`t_s`$ 继续传播。

因此，熵差负责回答“什么时候拆分”，但不直接回答“沿哪个状态方向拆分”。方向选择由协方差特征向量完成。

### 4. 存在过程噪声时的变化

若连续过程噪声协方差为 $`\boldsymbol{Q}`$，线性化协方差变为：

```math
\dot{\boldsymbol{P}}=\boldsymbol{F}\boldsymbol{P}+\boldsymbol{P}\boldsymbol{F}^{\mathsf T}+\boldsymbol{Q}.
```

此时：

```math
\dot H_{\mathrm{lin}}=\operatorname{tr}(\boldsymbol{F})+\frac12\operatorname{tr}(\boldsymbol{P}^{-1}\boldsymbol{Q}).
```

新增项依赖完整协方差，所以原文式（13）的标量捷径不再成立。论文指出，此时必须同时运行完整的线性化预测器和非线性预测器，分别用式（5）计算熵后再比较。

---

## 五、AEGIS 机制二：建立一维高斯拆分库

### 1. 拆分目标

先考虑标准高斯：

```math
p(x)=\mathcal N(x;0,1). \quad (15)
```

希望用 $`G`$ 个较小高斯近似：

```math
\widetilde p(x)=\sum_{i=1}^{G}\widetilde\alpha_i\,\mathcal N(x;\widetilde m_i,\widetilde\sigma^2). \quad (16)
```

拆分库被限制为同方差形式，即所有子分量使用相同的 $`\widetilde\sigma^2`$。

### 2. 优化准则

论文使用 beta divergence 衡量原高斯与拆分 GMM 之间的距离。在 $`\beta=2`$ 时，它等价于按比例缩放的 $`L_2`$ 距离，并且高斯与 GMM 之间的积分可以解析计算。

优化目标为原文式（17）：

```math
J=D_B^{(2)}(p\|\widetilde p)+\lambda_{\mathrm{pen}}\widetilde\sigma^2,qquad \sum_{i=1}^{G}\widetilde\alpha_i=1. \quad (17)
```

- 第一项要求拆分后的混合密度接近原高斯；
- 第二项鼓励每个子分量具有更小的协方差；
- $`\lambda_{\mathrm{pen}}`$ 是方差惩罚系数，不要与协方差特征值 $`\lambda_k`$ 混淆。

### 3. 三分量拆分库

论文表 1 给出的三分量库为：

```math
\widetilde{\boldsymbol\alpha}=[0.2252246249,\ 0.5495507502,\ 0.2252246249],
```

```math
\widetilde{\boldsymbol m}=[-1.0575154615,\ 0,\ 1.0575154615],\qquad \widetilde\sigma=0.6715662887.
```

参数采用 $`\beta=2`$、$`\lambda_{\mathrm{pen}}=0.001`$。

### 4. 五分量拆分库

论文表 2 给出的五分量库为：

```math
\widetilde{\boldsymbol\alpha}=[0.0763216491,\ 0.2474417860,\ 0.3524731300,\ 0.2474417860,\ 0.0763216491],
```

```math
\widetilde{\boldsymbol m}=[-1.6899729111,\ -0.8009283834,\ 0,\ 0.8009283834,\ 1.6899729111],\qquad \widetilde\sigma=0.4422555386.
```

参数采用 $`\beta=2`$、$`\lambda_{\mathrm{pen}}=0.0025`$。

五分量库更精细、子分量更窄，但每次触发会增加更多轨道积分。

---

## 六、从一维拆分推广到多维

设待拆分的父分量为：

```math
\alpha\,\mathcal N(\boldsymbol{x};\boldsymbol{m},\boldsymbol{P}).
```

希望替换成：

```math
\alpha\,\mathcal N(\boldsymbol{x};\boldsymbol{m},\boldsymbol{P})\approx\sum_{i=1}^{G}\alpha_i\,\mathcal N(\boldsymbol{x};\boldsymbol{m}_i,\boldsymbol{P}_i). \quad (18)
```

### 1. 选择拆分方向

对父协方差做谱分解：

```math
\boldsymbol{P}=\boldsymbol{V}\boldsymbol{\Lambda}\boldsymbol{V}^{\mathsf T},\qquad \boldsymbol{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_n).
```

$`\boldsymbol{v}_k`$ 是第 $`k`$ 个特征向量，$`\lambda_k`$ 是该方向的方差。选择第 $`k`$ 个主方向应用一维拆分库。

论文的轨道算例沿最大特征值对应的主方向拆分。一般框架也允许选择多个方向，但若沿 $`m`$ 个方向递归使用 $`G`$ 分量库，最终可能产生 $`G^m`$ 个分量。

### 2. 子分量参数

第 $`i`$ 个子分量的权重为：

```math
\boxed{\alpha_i=\widetilde\alpha_i\alpha}.
```

均值为：

```math
\boxed{\boldsymbol{m}_i=\boldsymbol{m}+\sqrt{\lambda_k}\,\widetilde m_i\boldsymbol{v}_k}.
```

协方差为：

```math
\boldsymbol{P}_i=\boldsymbol{V}\boldsymbol{\Lambda}_i\boldsymbol{V}^{\mathsf T},
```

```math
\boldsymbol{\Lambda}_i=\operatorname{diag}(\lambda_1,\ldots,\widetilde\sigma^2\lambda_k,\ldots,\lambda_n).
```

物理意义是：

- 沿第 $`k`$ 个方向移动子分量中心；
- 沿该方向缩小每个子分量的局部方差；
- 其他方向的局部方差保持不变；
- 子分量权重之和仍等于父分量权重。

对称拆分库满足加权中心偏移接近零，因此拆分后的混合均值保持在父均值附近；“子分量内部方差＋子分量均值之间的离散”共同重构父分量的不确定性。

> 拆分没有让系统凭空获得更多信息，也不应被解释为整体不确定性降低；它只是把一个大椭球改写为沿主方向排列的多个小椭球。

---

## 七、完整 AEGIS 传播算法

### 1. 为每个分量生成 sigma 点

对第 $`\ell`$ 个分量进行平方根分解：

```math
\boldsymbol{P}_{\ell,k-1}^{+}=\boldsymbol{S}_{\ell,k-1}\boldsymbol{S}_{\ell,k-1}^{\mathsf T}. \quad (19)
```

设 $`\boldsymbol{s}_{\ell,i,k-1}`$ 为 $`\boldsymbol S`$ 的第 $`i`$ 列。论文使用 $`K=2n`$ 个对称 sigma 点（原文式 20）：

```math
\boldsymbol{X}_{\ell,i,k-1}=\boldsymbol{m}_{\ell,k-1}^{+}+\sqrt n\,\boldsymbol{s}_{\ell,i,k-1},
```

```math
\boldsymbol{X}_{\ell,i+n,k-1}=\boldsymbol{m}_{\ell,k-1}^{+}-\sqrt n\,\boldsymbol{s}_{\ell,i,k-1},\qquad w_i=\frac{1}{2n}. \quad (20)
```

### 2. sigma 点通过完整动力学传播

```math
\dot{\boldsymbol{X}}_{\ell,i}(t)=\boldsymbol{f}(\boldsymbol{X}_{\ell,i}(t),t). \quad (21)
```

分量概率权重在传播区间内保持不变：

```math
\dot\alpha_\ell(t)=0.
```

### 3. 找到最早的熵差超阈值时刻

持续计算每个分量的：

```math
\Delta H_\ell(t)=|H_{\ell,\mathrm{nl}}(t)-H_{\ell,\mathrm{lin}}(t)|.
```

若某个分量首次在 $`t_s`$ 超过阈值，则所有分量先传播到 $`t_s`$，并由 sigma 点重构：

```math
\boldsymbol{m}_{\ell,s}=\sum_iw_i\boldsymbol{X}_{\ell,i,s},
```

```math
\boldsymbol{P}_{\ell,s}=\sum_iw_i(\boldsymbol{X}_{\ell,i,s}-\boldsymbol{m}_{\ell,s})(\boldsymbol{X}_{\ell,i,s}-\boldsymbol{m}_{\ell,s})^{\mathsf T}.
```

### 4. 仅拆分超阈值分量

若第 $`j`$ 个分量触发，则用 $`G`$ 个新分量替换它（原文式 22）：

```math
\alpha_{j,s}\mathcal N(\boldsymbol{x};\boldsymbol{m}_{j,s},\boldsymbol{P}_{j,s})\approx\sum_{r=1}^{G}\alpha_{r,s}\mathcal N(\boldsymbol{x};\boldsymbol{m}_{r,s},\boldsymbol{P}_{r,s}). \quad (22)
```

总分量数更新为：

```math
L\leftarrow L+G-1.
```

为新分量重新生成 sigma 点，从 $`t_s`$ 继续传播。若后来再次超阈值，就再次执行该过程，直到终止时刻 $`t_k`$。

### 5. 算法直觉

```math
\text{大分量上的强非线性}\rightarrow\text{多个小分量上的较弱局部非线性}.
```

AEGIS 不是消除动力学非线性，而是自适应缩小每个局部概率块，使低阶高斯传播在更小区域内重新变得合理。

---

## 八、如何评价传播结果：Likelihood Agreement Measure

论文用 Likelihood Agreement Measure（LAM）评价 GMM 与蒙特卡洛样本的符合程度：

```math
\mathcal L(p,q)=\int p(\boldsymbol{x})q(\boldsymbol{x})\,d\boldsymbol{x}. \quad (23)
```

把 $`K`$ 个等权蒙特卡洛样本写成 Dirac 混合：

```math
q(\boldsymbol{x})=\sum_{i=1}^{K}\gamma_i\delta(\boldsymbol{x}-\boldsymbol{\mu}_i),\qquad \gamma_i=\frac1K. \quad (24)
```

若预测 GMM 为：

```math
p(\boldsymbol{x})=\sum_{j=1}^{L}\alpha_j\mathcal N(\boldsymbol{x};\boldsymbol{m}_j,\boldsymbol{P}_j), \quad (25)
```

则 LAM 可直接计算为：

```math
\mathcal L(p,q)=\sum_{i=1}^{K}\sum_{j=1}^{L}\gamma_i\alpha_j\mathcal N(\boldsymbol{\mu}_i;\boldsymbol{m}_j,\boldsymbol{P}_j). \quad (26)
```

LAM 越大，说明蒙特卡洛样本落在预测密度高概率区域的程度越高。它不是归一化距离，因此论文主要使用相对或归一化后的 LAM 比较不同方法。

---

## 九、数值算例

### 1. 偏心高地球轨道

算例采用二维两体动力学：

- 半长轴：$`35{,}000\ \mathrm{km}`$；
- 偏心率：$`0.2`$；
- 近地点幅角和初始平近点角：$`0^\circ`$；
- 初始位置标准差：$`1\ \mathrm{km}`$；
- 初始速度标准差：$`1\ \mathrm{m/s}`$；
- 传播时长：两个轨道周期；
- 蒙特卡洛样本数：$`1000`$；
- 拆分库：五分量；
- 熵差阈值：$`\Delta H=0.003H_0`$。

两体状态空间的雅可比迹为零，因此线性参考熵保持常数。约传播 12 小时后，AEGIS 首次检测到非线性并开始拆分。论文图 2–4 表明：

- UKF 的 LAM 随传播迅速低于 AEGIS；
- AEGIS 的位置和速度边缘密度能跟随蒙特卡洛样本的弯曲结构；
- 单高斯 UKF 无法表示这种曲率；
- 虽然只沿协方差最大特征值方向拆分，但传播积累的相关性使子分量最终分布到多个状态变量中。

### 2. 含大气阻力的圆形低地球轨道

算例采用二维引力与指数大气模型：

- 圆轨道高度：$`225\ \mathrm{km}`$；
- 弹道系数参数：$`\beta=1.4`$；
- 初始 $`x,y`$ 位置标准差：$`1.3\ \mathrm{km}`$、$`0.5\ \mathrm{km}`$；
- 初始 $`u,v`$ 速度标准差：$`2.5\ \mathrm{m/s}`$、$`5\ \mathrm{m/s}`$；
- 传播时长：两个轨道周期；
- 蒙特卡洛样本数：$`1000`$；
- 分别测试三分量与五分量拆分库；
- 熵差阈值：$`\Delta H=0.001H_0`$。

该系统雅可比迹不为零，线性参考熵随阻力变化。约传播 30 分钟后，AEGIS 首次触发拆分。论文图 6–9 表明：

- 三分量和五分量 AEGIS 都优于单高斯 UKF；
- 五分量方案的 LAM 优于三分量方案；
- 两种 AEGIS 的边缘等高线都能较好表达蒙特卡洛样本曲率；
- 三分量方案的单次拆分更便宜，但算例中两种方案最终分量数接近，因此五分量方案没有表现出数量级上的额外计算负担。

### 3. 计算量

论文中每个分量都运行一次 UKF 型 sigma 点传播，因此每一步的计算量约为：

```math
\mathcal O(L\times\text{一次UKF传播}).
```

AEGIS 用“只在需要时拆分”换取更准确的密度形状，但原论文没有在传播阶段引入系统性的剪枝与合并机制，长期传播时仍需要控制分量爆炸。

---

## 十、核心认识与工程注意事项

### 1. 熵差是局部非线性指标

AEGIS 对每个 GMM 分量独立监测熵差。它比较的是同一个局部高斯在“线性化预测”和“非线性 sigma 点预测”下的不确定性体积变化，而不是直接计算整个 GMM 的精确混合熵。

### 2. 拆分是重表示，不是信息更新

父权重通过 $`\alpha_i=\widetilde\alpha_i\alpha`$ 分配给子分量，总概率质量保持不变。拆分不会使真实不确定性凭空下降，也不会引入新的观测信息。

### 3. 熵只决定何时拆分

熵差是标量，不能说明哪个方向最非线性。原论文通过协方差主方向完成拆分，并在算例中选择最大特征值方向。这是“最大离散方向”，不一定永远等同于“最强动力学曲率方向”。

### 4. 初始协方差必须正定

AEGIS 要对协方差做 Cholesky 或谱分解。若某些状态被完全固定、协方差降秩，sigma 点生成和熵的 $`\log|\boldsymbol P|`$ 都会失败。工程上需要保留真实的小不确定性或采用正定正则化。

### 5. 阈值需要在固定坐标尺度下标定

论文使用 $`0.003H_0`$ 和 $`0.001H_0`$ 作为算例阈值，但这不是适用于所有轨道的通用值。阈值应结合：

- 状态无量纲化方式；
- 允许的概率密度误差；
- 最大分量数与计算预算；
- 蒙特卡洛或高保真星历验证。

### 6. 存在过程噪声时不能使用标量捷径

若加入未建模加速度、机动或其他过程噪声，必须传播完整的线性化与非线性协方差；否则过程噪声造成的熵增长可能被误判为动力学非线性。

### 7. 需要配套的复杂度控制

实际滤波系统还应增加：

- 低权重分量剪枝；
- 相近分量合并；
- 最大拆分深度；
- 最大总分量数；
- 拆分后的冷却时间或滞回阈值，避免短时间反复拆分。

---

## 十一、对地月与 CRTBP 不确定性传播的启示

标准六维 CR3BP 一阶状态形式通常满足：

```math
\operatorname{tr}(\boldsymbol F)=0.
```

因此在无过程噪声情况下：

```math
H_{\mathrm{lin}}(t)=H(t_0).
```

AEGIS 检测可简化为监测：

```math
|H_{\mathrm{nl}}(t)-H(t_0)|>\Delta H_{\mathrm{th}}.
```

对于由距离—距离率可容许域构造的初始 GMM，可以对每个网格分量独立应用该判据：

```math
\text{AR初始GMM}\rightarrow\text{CRTBP逐分量传播}\rightarrow\text{熵差检测}\rightarrow\text{必要时沿主方向拆分}\rightarrow\text{剪枝与合并}.
```

但应注意，CRTBP 的相空间体积守恒并不意味着高斯近似始终准确。真实密度可以在保持相空间体积的同时被拉伸和弯曲；sigma 点重构的高斯协方差行列式变化正是在提示“用单个椭球包住弯曲分布”所产生的近似误差。

---

## 十二、自测问题

1. 为什么固定分量数的 GMM 在长期非线性传播中可能失败？
2. 高斯微分熵为什么只依赖协方差而不依赖均值？
3. 无过程噪声时，为什么线性化熵导数可以化成 $`\operatorname{tr}(\boldsymbol F)`$？
4. AEGIS 中的线性熵和非线性熵分别如何获得？
5. 熵差超过阈值意味着什么？
6. 熵差能否直接确定拆分方向？
7. 一维拆分库中的方差惩罚项有什么作用？
8. 多维高斯的拆分方向如何定义？
9. 为什么拆分不会降低真实整体不确定性？
10. 若沿 $`m`$ 个方向使用 $`G`$ 分量库，可能生成多少个分量？
11. 存在过程噪声时，为什么原文式（13）不能直接使用？
12. LAM 的数值越大表示什么？
13. 为什么 AEGIS 协方差必须正定？
14. 为什么在实际系统中还需要剪枝和合并？

---

## 简短答案

1. 单个分量会覆盖过大的强非线性区域，传播后真实密度不再近似高斯。
2. 高斯的平移不改变形状和体积，信息体积由 $`|\boldsymbol P|`$ 决定。
3. 将 $`\dot{\boldsymbol P}=\boldsymbol F\boldsymbol P+\boldsymbol P\boldsymbol F^{\mathsf T}`$ 代入熵导数并利用迹的循环性质。
4. 线性熵积分标量方程，非线性熵由 sigma 点重构协方差后代入高斯熵公式。
5. 当前局部高斯的非线性传播已明显偏离线性化基准，需要提高表示分辨率。
6. 不能；方向需由协方差特征向量或其他方向指标另行选择。
7. 促使子分量更窄，使拆分真正降低局部覆盖尺度。
8. 对父协方差做谱分解，沿选定特征向量应用一维拆分库。
9. 父权重守恒，子分量中心之间的离散继续承载父分量的不确定性。
10. 最多 $`G^m`$ 个。
11. 过程噪声引入依赖 $`\boldsymbol P^{-1}\boldsymbol Q`$ 的额外熵增项，必须传播完整协方差。
12. 预测 GMM 与蒙特卡洛样本代表的分布重合程度更高。
13. sigma 点平方根分解和 $`\log|\boldsymbol P|`$ 都要求非奇异正定协方差。
14. 递归拆分会使计算量快速增长，观测更新后还会产生低权重或冗余分量。

---

## 参考文献

```bibtex
@article{DeMars2013entropy,
  author  = {DeMars, Kyle J. and Bishop, Robert H. and Jah, Moriba K.},
  title   = {Entropy-Based Approach for Uncertainty Propagation of Nonlinear Dynamical Systems},
  journal = {Journal of Guidance, Control, and Dynamics},
  volume  = {36},
  number  = {4},
  pages   = {1047--1057},
  year    = {2013},
  doi     = {10.2514/1.58987}
}
```
