# HNEKF：高阶数值扩展卡尔曼滤波详解

> 高阶数值扩展卡尔曼滤波（Higher-order Numerical Extended Kalman Filter, HNEKF）在标准 EKF 框架下引入状态转移张量，用于强非线性系统（如地月 CR3BP）的状态估计。

## 1. 非线性系统与解流

考虑非线性动力学系统：

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), t), \qquad
\mathbf{x}(t) = \phi(t; \mathbf{x}_0, t_0)
$$

- $\mathbf{x}(t)$：系统状态（如位置 + 速度）
- $\mathbf{f}$：非线性向量场（例如圆限制性三体问题）
- $\phi$：**解流**，表示从初始状态 $\mathbf{x}_0$ 出发经过时间 $t - t_0$ 后的状态

**小扰动传播**：要分析初始小偏差随时间的变化，需要状态对初值的灵敏度（导数）。

---

## 2. 一阶灵敏度：状态转移矩阵 (STM)

一阶状态转移张量定义为：

$$
\phi_{(t,t_0)}^{i,j} = \frac{\partial \phi^i(t;\mathbf{x}_0,t_0)}{\partial x_0^j}
$$

- $\phi^i$：解流的第 $i$ 个分量
- $x_0^j$：初始状态的第 $j$ 个分量
- 该偏导数描述了初始偏差在 $j$ 方向对最终状态 $i$ 方向的影响
- 所有 $i,j$ 构成**状态转移矩阵** $\Phi(t,t_0)$

---

## 3. 高阶灵敏度：$p$ 阶状态转移张量

为捕捉非线性耦合，引入高阶偏导：

$$
\phi_{(t,t_0)}^{i,\gamma_1,\dots,\gamma_p} =
\frac{\partial^p \phi^i(t;\mathbf{x}_0,t_0)}{\partial x_0^{\gamma_1} \dots \partial x_0^{\gamma_p}}
$$

- 这是一个 $p+1$ 维张量
- $p=2$ 时，$\phi^{i,ab}$ 表示初始 $a$ 和 $b$ 方向同时有偏差对最终 $i$ 分量的二阶影响
- 一阶张量描述线性变换，二阶张量描述曲率（弯曲效应）

---

## 4. 状态转移张量的传播微分方程

一阶和二阶张量满足以下微分方程（爱因斯坦求和约定）：

$$
\begin{aligned}
\dot{\phi}^{i,a} &= f^{i,\alpha}\,\phi^{\alpha,a} \\
\dot{\phi}^{i,ab} &= f^{i,\alpha\beta}\,\phi^{\alpha,a}\phi^{\beta,b} + f^{i,\alpha}\,\phi^{\alpha,ab}
\end{aligned}
$$

其中：

- $f^{i,\alpha} = \dfrac{\partial f^i}{\partial x^\alpha}$（雅可比矩阵元素）
- $f^{i,\alpha\beta} = \dfrac{\partial^2 f^i}{\partial x^\alpha \partial x^\beta}$（海森张量）

**物理含义**：

- 第一式：一阶张量变化率 = 雅可比 × 一阶张量（线性传播）
- 第二式：二阶张量变化率 = 雅可比 × 二阶张量（线性部分）+ 海森 × 两个一阶张量的乘积（非线性耦合）

这些方程可与轨道积分同时数值求解。

---

## 5. 初始条件

在 $t = t_0$ 时：

$$
\begin{aligned}
\phi^{i,j}(t_0,t_0) &= \delta^{ij} \quad (\text{Kronecker delta}) \\
\phi^{i,jk}(t_0,t_0) &= 0 \\
\phi^{i,jkl}(t_0,t_0) &= 0 \quad (\text{所有 } p \ge 2)
\end{aligned}
$$

- 一阶张量初始为单位矩阵
- 高阶张量初始为零（初始时刻无弯曲累积）

---

## 6. 用张量近似状态偏差

将初始小偏差 $\delta\mathbf{x}_0$ 映射到最终偏差 $\delta\mathbf{x}(t)$：

$$
\delta x^i(t) \approx \sum_{p=1}^{m} \frac{1}{p!}\,
\phi_{(t,t_0)}^{i,\gamma_1,\dots,\gamma_p}\,
\delta x_0^{\gamma_1} \cdots \delta x_0^{\gamma_p}
$$

- $p=1$ 项：线性（STM）传播
- $p=2$ 项：二次耦合项
- 在强非线性环境（如地月系 CR3BP）中，必须保留高阶项

---

## 7. HNEKF 滤波结构

HNEKF 在标准 EKF 的预测和更新步骤中引入高阶张量，分为：

- **时间更新（预测）**：传播状态均值和协方差
- **测量更新（修正）**：利用测量值修正

### 7.1 预测步：均值传播

公式 (14)：

$$
(m_{k+1}^{-})^{i} = \phi^{i}(t_{k+1};\mathbf{m}_k^{+},t_k) + \delta m_{k+1}^{i}(\delta\mathbf{x}_k)
$$

- 第一项：将后验均值 $\mathbf{m}_k^{+}$ 经非线性动力学精确积分
- 第二项：高阶修正项，补偿非线性引起的均值漂移

修正项具体形式 (15)：

$$
\delta m_{k+1}^{i} = \sum_{p=1}^{m} \frac{1}{p!}\,
\phi_{(t_{k+1},t_k)}^{i\gamma_1\cdots\gamma_p}\;
\mathbb{E}\left[\delta x_k^{\gamma_1}\cdots\delta x_k^{\gamma_p}\right]
$$

- $\phi$： $p$ 阶状态转移张量
- $\mathbb{E}[\cdot]$：对当前估计误差 $\delta\mathbf{x}_k = \mathbf{x}_k - \mathbf{m}_k^{+}$ 的各阶矩求期望

若系统线性，高阶张量为零，退化为标准 EKF 均值积分。

### 7.2 预测步：协方差传播

公式 (16)：

$$
\begin{aligned}
(P_{k+1}^{-})^{ij} = &\sum_{p=1}^{m}\sum_{q=1}^{m} \frac{1}{p!\,q!}\,
\phi^{i\gamma_1\cdots\gamma_p}\,\phi^{j\zeta_1\cdots\zeta_q}\,
\mathbb{E}\left[\delta x_k^{\gamma_1}\cdots\delta x_k^{\gamma_p}\,
\delta x_k^{\zeta_1}\cdots\delta x_k^{\zeta_q}\right] \\
&- \delta m_{k+1}^{i}\,\delta m_{k+1}^{j} + Q_k^{ij}
\end{aligned}
$$

- 第一项：将初始误差的高阶矩映射到最终误差的非中心二阶矩
- 第二项：减去均值乘积，得到中心协方差
- 第三项：过程噪声 $Q_k$

标准 EKF 仅用一阶项且无均值修正，得到 $P_{k+1}^{-} = \Phi P_k^{+} \Phi^\top + Q$。HNEKF 的高阶项可捕获协方差的非线性形变。

### 7.3 高斯假设下的矩计算

假设 $\delta\mathbf{x}_k \sim \mathcal{N}(0, P_k^{+})$，则：

- $\mathbb{E}[\delta x^a] = 0$
- $\mathbb{E}[\delta x^a \delta x^b] = P^{ab}$
- 三阶矩 = 0
- 四阶矩：

$$
\mathbb{E}[\delta x^a \delta x^b \delta x^c \delta x^d] =
P^{ab}P^{cd} + P^{ac}P^{bd} + P^{ad}P^{bc}
$$

所有期望均可由当前协方差 $P_k^{+}$ 表示。

### 7.4 测量预测

公式 (22)：

$$
(n_{k+1}^{-})^{i} = h^{i}\!\left(\phi(t_{k+1};\mathbf{m}_k^{+},t_k)\right) + (\delta n_{k+1}^{-})^{i}
$$

- 第一项：将预测状态均值代入测量函数 $h$
- 第二项：高阶修正项

$$
(\delta n_{k+1}^{-})^{i} = \sum_{p=1}^{m} \frac{1}{p!}\,
h_{(t_{k+1},t_k)}^{i,\gamma_1\cdots\gamma_p}\;
\mathbb{E}\left[\delta x_k^{\gamma_1}\cdots\delta x_k^{\gamma_p}\right]
$$

其中 $h_{(t_{k+1},t_k)}^{i,\gamma_1\cdots\gamma_p}$ 是测量函数与流的复合高阶偏导（公式 23）：

- 一阶： $h^{i,a} = h_{k+1}^{i,\alpha}\,\phi^{\alpha,a}$
- 二阶： $h^{i,ab} = h_{k+1}^{i,\alpha}\,\phi^{\alpha,ab} + h_{k+1}^{i,\alpha\beta}\,\phi^{\alpha,a}\phi^{\beta,b}$

这里 $h_{k+1}^{i,\alpha} = \dfrac{\partial h^{i}}{\partial x^{\alpha}}$ 在预测状态处取值。

### 7.5 测量协方差与互协方差

公式 (26) 定义了测量残差协方差 $P_{k+1}^{zz}$ 和状态-测量互协方差 $P_{k+1}^{xz}$。结构与状态协方差类似，均包含高阶张量映射、均值修正，并加上测量噪声 $R_{k+1}$。

若测量模型为线性 $h(\mathbf{x}) = H\mathbf{x}$，则高阶项消失，简化为标准 EKF 形式（公式 27）。

### 7.6 卡尔曼增益与状态更新

无论测量模型如何，更新公式与标准卡尔曼滤波相同：

$$
\begin{aligned}
\mathbf{K}_{k+1} &= \mathbf{P}_{k+1}^{\mathbf{xz}} (\mathbf{P}_{k+1}^{\mathbf{zz}})^{-1} \\
\mathbf{m}_{k+1}^{+} &= \mathbf{m}_{k+1}^{-} + \mathbf{K}_{k+1} (\mathbf{z}_{k+1} - \mathbf{n}_{k+1}^{-}) \\
\mathbf{P}_{k+1}^{+} &= \mathbf{P}_{k+1}^{-} - \mathbf{K}_{k+1} \mathbf{P}_{k+1}^{\mathbf{zz}} \mathbf{K}_{k+1}^{\top}
\end{aligned}
$$

其中所有预测量 ： 

$$
\mathbf{m}_{k+1}^{-},\;
\mathbf{P}_{k+1}^{-},\;
\mathbf{n}_{k+1}^{-},\;
\mathbf{P}_{k+1}^{\mathbf{zz}},\;
\mathbf{P}_{k+1}^{\mathbf{xz}}
$$ 

均来自前述高阶修正。

---

## 8. 关键区别总结

| 步骤 | 标准 EKF | HNEKF（二阶） |
| --- | --- | --- |
| 状态预测 | 只积分均值 | 均值 + 二阶修正项 |
| 协方差预测 | $\Phi P \Phi^\top + Q$ | 含二阶张量、四阶矩、减均值修正 |
| 测量预测 | $h(\mathbf{m}^-)$ | $h(\mathbf{m}^-) +$ 二阶修正项 |
| 测量协方差 | $H P^- H^\top + R$ | 含二阶张量、四阶矩、减均值修正 |

在强非线性系统（如地月系三体问题）中，HNEKF 能保持稳定收敛，而标准 EKF 容易发散。论文仿真表明：雷达测距/测速的强非线性下，只有包含二阶测量项的 HNEKF 才能成功估计。

---

## 附：关于记号 $f^{i,\alpha}$ 的说明

$f^{i,\alpha}$ 本质上是对状态变量 $x^\alpha$ 的偏导：

$$
f^{i,\alpha} \equiv \frac{\partial f^i}{\partial x^\alpha}
$$

在链式法则推导中，由于解流 $\phi$ 表示当前状态，也会写成 $\dfrac{\partial f^i}{\partial \phi^\alpha}$，两者含义相同，只是强调偏导数沿真实轨迹求值。这种简化记号在高阶张量推导中非常便捷。

**例子**：若 $\dot{x} = f(x) = x^2$，则 $f^{,1} = 2x$，一阶张量传播方程为 $\dot{\phi}^{,1} = 2\phi \cdot \phi^{,1}$。
