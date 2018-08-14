---
title: "Logistic Regression"
layout: page
date: 2018-06-29 00:00
---

[TOC]

## Logistic Regression Problem

一个心脏病预测的问题：根据患者的年龄、血压、体重等信息，来预测患者是否会有心脏病。很明显这是一个二分类问题，其输出y只有{-1,1}两种情况。
但是，如果我们想知道的不是患者有没有心脏病，而是到底患者有多大的几率是心脏病。这表示，我们更关心的是目标函数的值（分布在0,1之间），表示是正类的概率（正类表示是心脏病）。这跟我们原来讨论的二分类问题不太一样，我们把这个问题称为软性二分类问题（’soft’ binary classification）。这个值越接近1，表示正类的可能性越大；越接近0，表示负类的可能性越大。
目标函数target function: $$f(x)=P(+1|x)\in [0,1]$$
对于软性二分类问题，理想的label为概率值，是分布在[0,1]之间的具体值，但是实际数据中label是0或者1，我们可以把实际中的数据看成是理想数据加上了噪声的影响。

<img src="/wiki/static/images/logistic_regression/noise_data.png" alt="noise_data"/>

和hard binary classfication相比，使用相同的数据集，但学到不同的target function。

如果目标函数是$f(x)=P(+1|x)∈[0,1]$的话，我们如何找到一个好的Hypothesis跟这个目标函数很接近呢？
首先，根据我们之前的做法，对所有的特征值进行加权处理，计算的结果s，我们称之为 risk score：

$$
for\  \mathbf{x}=(x_{0},x_{1},x_{2},\cdots ,x_{d}) ,   s=\sum_{i=0}^{d}w_{i}x_{i}
$$
但是特征加权和s∈(−∞,+∞)，如何将s值限定在[0,1]之间呢？一个方法是使用sigmoid Function，记为θ(s)

<img src="/wiki/static/images/logistic_regression/sigmoid.png" alt="sigmoid function"/>

Sigmoid Function函数记为$\theta (s)=\frac{1}{1+e^{-s}}$，满足$\theta (-\infty)=0,\theta (0)=0.5,\theta (+\infty)=1$。这个函数是平滑的、单调的S型函数。则对于逻辑回归问题，hypothesis就是这样的形式：

$$
h(\mathbf{x})=\frac{1}{1+e^{-\mathbf{w}^{T}\mathbf{x}}}
$$
那我们的目标就是求出这个预测函数h(x)，使它接近目标函数f(x)

## Logistic Regression Error

现在我们将Logistic Regression与之前讲的Linear Classification、Linear Regression做个比较：

<img src="/wiki/static/images/logistic_regression/lr_compare.png" alt="lr_compare"/>

这三个线性模型都会用到线性scoring function $s=\mathbf{w}^{T}\mathbf{x}$。linear classification的误差使用的是0/1 error，linear regression的误差使用的是squared error，那么logistic regression的误差该如何定义呢？

先介绍一下“似然性”的概念。目标函数$f(x)=P(+1|x)$，如果我们找到了hypothesis很接近target function， 也就是说，在所有的Hypothesis集合中找到一个hypothesis与target function最接近，那么由该hypothesis生成数据集D的似然值应该与由target function 生成数据集D的概率值很接近。

<img src="/wiki/static/images/logistic_regression/max_likelihood.png" alt="max_likelihood"/>

logistic function：$h(\mathbf{x})=\theta (\mathbf{w}^{T}\mathbf{x})$ 满足一个性质：$1−h(x)=h(−x)$， 因此似然函数可以写成：

$$
likelihood(h)=P(\mathbf{x}\_{1})h(+\mathbf{x}\_{1})\times P(\mathbf{x}\_{2})h(-\mathbf{x}\_{2})\times \cdots \times P(\mathbf{x}\_{N})h(+\mathbf{x}\_{N})
$$
因为$P(\mathbf{x}\_{n})$对所有的hypothesis来说，都是一样的，所以我们可以忽略它，那么可以得到$likelihood(h)$正比于所有$h(y_{n}\mathbf{x}\_{n})$的乘积，通常情况下target function $f$ 生成数据集D的probability很大，因此我们的目标就是让所有$h(y_{n}\mathbf{x}\_{n})$的乘积值最大化。
$$
\max_{h}\ likelihood(logistic \ h)\propto \prod_{n=1}^{N} h(y_{n}\mathbf{x}_{n})
$$

将$h(\mathbf{x})=\theta (\mathbf{w}^{T}\mathbf{x})$代入

$$
\max_{\mathbf{w}}\ likelihood(logistic \ h)\propto \prod_{n=1}^{N} \theta (y_{n}\mathbf{w^{T}}\mathbf{x}_{n})
$$

为了把连乘问题简化计算，我们可以引入ln操作，让连乘转化为连加

$$
\max_{\mathbf{w}}ln \prod_{n=1}^{N} \theta (y_{n}\mathbf{w^{T}}\mathbf{x}\_{n})=\max_{\mathbf{w}}\sum_{n=1}^{N}ln\theta (y_{n}\mathbf{w^{T}}\mathbf{x}_{n})
$$

接着，我们将maximize问题转化为minimize问题，并引入平均数操作1/N：

$$
\min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}-ln\theta (y_{n}\mathbf{w^{T}}\mathbf{x}_{n})
$$

将logistic function的表达式$\theta (s)=\frac{1}{1+e^{-s}}$代入得到

$$
\min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}}) \Rightarrow \min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}err(\mathbf{w},\mathbf{x_{n}},y_{n})
$$
至此，我们得到了logistic regression的err function，称之为cross-entropy error交叉熵损失函数。

## Gradient of Logistic Regression Error

我们已经推导了$E_{in}$的表达式，那接下来的问题就是如何找到合适的向量$\mathbf{w}$，让$E_{in}$最小。
$$
\min_{\mathbf{w}}E_{in}(\mathbf{w})=\min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}})
$$

Logistic Regression的$E_{in}$是关于$\mathbf{w}$的连续、可微、二次可微的凸曲线（开口向上），根据之前Linear Regression的思路，我们只要计算$E_{in}$的梯度为零时的$\mathbf{w}$，即为最优解。

<img src="/wiki/static/images/logistic_regression/lr_Ein.png" alt="lr_Ein"/>

根据链式求导法则，计算$E_{in}$梯度
$$
\begin{equation}
\begin{aligned}
\frac{\partial E_{in}(\mathbf{w})}{\partial w_{i}}&=\frac{\partial \frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}})}{\partial w_{i}}\\\\
&= \frac{1}{N}\sum_{n=1}^{N}\frac{\partial ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}})}{\partial w_{i}}\\\\
&=    \frac{1}{N}\sum_{n=1}^{N} \frac{1}{1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}}}\frac{\partial (1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}})}{\partial w_{i}}   \\\\
&=    \frac{1}{N}\sum_{n=1}^{N}(-y_{n}x_{n,i}) \frac{1}{1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}}} e^{-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}}\\\\
&= \frac{1}{N}\sum_{n=1}^{N}(-y_{n}x_{n,i})\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n})
\end{aligned}
\end{equation}
$$
写成向量的形式
$$
\nabla E_{in}(\mathbf{w})=\frac{1}{N}\sum_{n=1}^{N}\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n})(-y_{n}\mathbf{x_{n}})
$$
为了计算$E_{in}(\mathbf{w})$最小值，就要找到让$\nabla E_{in}(\mathbf{w})$等于0的位置。
上式中，$\nabla E_{in}(\mathbf{w})$可以看成是对$(-y_{n}\mathbf{x_{n}})$的加权平均，权重为$\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n})$，要想使加权和为0，一种情况是所有的权重$\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n})$都是0，那么可以保证$\nabla E_{in}(\mathbf{w})$为0，$\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n})$是sigmoid function，根据其特性，只要让$-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}$≪0，即$y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}$≫0。$y_{n}\mathbf{w^{T}}\mathbf{x}\_{n}$≫0表示对于所有的点，$y_{n}$和$\mathbf{w^{T}}\mathbf{x}\_{n}$都是同号的，这表示数据集D必须是全部线性可分的才能成立。
但是实际情况不可能保证数据集线性可分，在非线性可分的情况下，只能通过使加权和为零，来求解w。这种情况没有closed-form解，与Linear Regression不同，只能用迭代方法求解。
先来回顾一下PLA算法，对误分类点，更新权重$(\mathbf{w})$，对于正确分类点，则不更新，PLA的迭代优化过程表示如下：
$$
\begin{equation}
\begin{aligned}
\mathbf{w}\_{t+1}=&\mathbf{w}\_{t}+ \left [sign(\mathbf{w}\_{t}^{T})\neq y_{n}   \right ]y_{n}\mathbf{x}\_{n}\\
=&\mathbf{w}\_{t}+ \underbrace{1}\_{\mathbf{\eta }}\cdot \underbrace{\left [sign(\mathbf{w}\_{t}^{T})\neq y_{n}   \right ]y_{n}\mathbf{x}\_{n}}_{\mathbf{\nu }}
\end{aligned}
\end{equation}
$$

上式可以这么理解，w每次更新包含两个内容：一个是每次更新的方向$y_{n}\mathbf{x}_{n}$，用$\mathbf{\nu }$表示，另一个是每次更新的步长$\mathbf{\eta }$。参数$(\mathbf{\nu },\mathbf{\eta })$和终止条件决定了我们的迭代优化算法。

## Gradient Descent

迭代优化让每次w都有更新：
$$
for\ t=0,1,2\cdots  , \mathbf{w}\_{t+1}=\mathbf{w}\_{t}+\mathbf{\eta }\mathbf{\nu }
$$
我们把$E_{in}$曲线看做是一个山谷的话，要求$E_{in}$最小，即可比作下山的过程。整个下山过程由两个因素影响：一个是下山的单位方向v；另外一个是下山的步长η。
<img src="/wiki/static/images/logistic_regression/downhill.png" alt="downhill"/>

利用微分思想和线性近似，假设每次下山我们只前进一小步，即η很小，那么根据泰勒Taylor一阶展开，可以得到：

$$
E_{in}(\mathbf{w}\_{t+1})=E_{in}(\mathbf{w}\_{t}+\eta\boldsymbol{\nu } )=E_{in}(\mathbf{w}\_{t})+\eta\boldsymbol{\nu }^{T}
\nabla E_{in}(\mathbf{w}\_{t})
$$

迭代的目的是让$E_{in}$越来越小，即让$E_{in}(\mathbf{w}\_{t+1}) < E_{in}(\mathbf{w}\_{t})$。η是标量，因为如果两个向量方向相反的话，那么他们的内积最小（为负），也就是说如果方向v与梯度$\nabla E_{in}(\mathbf{w}\_{t})$反向的话，那么就能保证每次迭代$E_{in}(\mathbf{w}\_{t+1}) < E_{in}(\mathbf{w}\_{t})$都成立。则我们令下降方向v为：

$$
\boldsymbol{\nu} =-\frac{\nabla E_{in}(\mathbf{w}\_{t})}{\left \| \nabla E_{in}(\mathbf{w}\_{t}) \right \|}
$$
v是单位向量，v每次都是沿着梯度的反方向走，这种方法称为梯度下降（gradient descent）算法。那么每次迭代公式就可以写成：

$$
\mathbf{w}\_{t+1}=\mathbf{w}\_{t}+-\eta \frac{\nabla E_{in}(\mathbf{w}\_{t})}{\left \| \nabla E_{in}(\mathbf{w}\_{t}) \right \|}
$$

下面讨论一下$\eta$的大小对迭代优化的影响：$\eta$如果太小的话，那么下降的速度就会很慢；$\eta$如果太大的话，那么之前利用Taylor展开的方法就不准了，造成下降很不稳定，甚至会上升。因此，$\eta$应该选择合适的值，一种方法是在梯度较小的时候，选择小的$\eta$，梯度较大的时候，选择大的$\eta$，即$\eta$正比于$\left \| \nabla E_{in}(\mathbf{w}\_{t}) \right \|$。这样保证了能够快速、稳定地得到最小值$E_{in}(\mathbf{w})$。
<img src="/wiki/static/images/logistic_regression/eta.png" alt="eta"/>

对学习速率$\eta$做个更修正，梯度下降算法的迭代公式可以写成：

$$
\mathbf{w}\_{t+1}=\mathbf{w}\_{t}+-{\eta}'\nabla E_{in}(\mathbf{w}\_{t})
$$
其中：

$$
{\eta}'=\frac{-\eta }{\left \| \nabla E_{in}(\mathbf{w}\_{t}) \right \|}
$$
总结一下基于梯度下降的Logistic Regression算法步骤如下：

1. 初始化$\mathbf{w}_{0}$
2. 计算梯度$\nabla E_{in}(\mathbf{w})=\frac{1}{N}\sum_{n=1}^{N}\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}\_{n})(-y_{n}\mathbf{x\_{n}})$
3. 迭代更新$\mathbf{w}\_{t+1}=\mathbf{w}\_{t}+-{\eta}'\nabla E_{in}(\mathbf{w}\_{t})$
4. 满足$\nabla E_{in}(\mathbf{w}\_{t})$≈0或者达到迭代次数，迭代结束

## 总结

首先，从逻辑回归的问题出发，将$P(+1|x)$作为目标函数，将$\theta (\mathbf{w}^{T}\mathbf{x})$作为hypothesis。接着，根据极大似然准则定义了logistic regression的err function，称之为cross-entropy error。然后，我们计算logistic regression error的梯度，最后，通过梯度下降算法，计算最优$\mathbf{w}_{t}$

## 思考
Q：如何证明交叉熵损失函数$\frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}})$是关于$\mathbf{w}$的的凸函数？
A：机器学习基石视频里给出了思路：求出二次微分的矩阵，该矩阵是正定的，则说明函数是convex的。
**需要自己证明一下！！！**

Q：为什么logistic regression的误差不用平方损失函数？

A：它会导致损失函数是一个关于参数向量 的非凸函数，而用对数损失函数就没有这种问题。凸函数的性质为我们后面求解参数向量 提供了极大便利，非凸函数有很多局部最优解，不利于求解 的计算过程。

Q：有些教材里交叉熵损失函数是这样的形式：L(ŷ ,y)=−(ylog ŷ +(1−y)log (1−ŷ ))  这是因为输出y的定义是{0,1}，而不是{-1,1} 因此可以写成这样的形式，而不能写成$h(y_{n}\mathbf{x}_{n})$的形式

## 个人总结

逻辑斯蒂回归用来处理二分类问题，大多数资料的推导过程中是设定类别$y_{i}\in  \{ 0,1  \}$的，也有的教材设定$y_i \in  \{ -1,1  \}$ 

1. Logistic的假设函数为$h_{\theta}(\mathbf{x})=\frac{1}{1+e^{-\mathbf{\theta}^{T}\mathbf{x}}}$，函数输出是介于（0，1）之间的，也就表明了属于某一类别的**概率**。对于输入x分类结果为类别1和类别0的概率分别为： 

$$
P(y=1|x;\theta)=h_{\theta}(x)\\
P(y=0|x;\theta)=1-h_{\theta}(x)\\
$$

2. 有了假设函数后，通过极大似然估计的方法得到损失函数：

   假设各个样本独立，真实的目标函数$f$(即类别1的概率函数)生成数据集$D$的概率值为：
   $$
   probability(f)=P(\mathbf{x}\_{1})f(\mathbf{x}\_{1})\times P(\mathbf{x}\_{2})(1-f(\mathbf{x}\_{2}))\times \cdots \times P(\mathbf{x}\_{N})f(\mathbf{x}\_{N})
   $$
   上式中$y_1=1,y_2=0,y_N=1...$
   假设函数$h_{\theta}(\mathbf{x})$生成数据集D的似然值为：
   $$
   likelihood(h)=P(\mathbf{x}\_{1})h(\mathbf{x}\_{1})\times P(\mathbf{x}\_{2})(1-h(\mathbf{x}\_{2}))\times \cdots \times P(\mathbf{x}\_{N})h(\mathbf{x}\_{N})
   $$
   如果假设函数与真实目标函数十分接近的话，那么似然值与目标函数产生数据集$D$的概率值也应该十分接近，

   通常情况下target function $f$ 生成数据集$D$的probability很大，因此我们的目标就是让似然值likelihood(h)最大。

   因为$P(\mathbf{x}\_{n})$对所有的hypothesis来说，都是一样的，所以我们可以忽略它，那么可以得到$likelihood(h)$正比于所有$h(\mathbf{x}\_{n})(y_n=1)$和$1-h(\mathbf{x}\_{n})(y_n=0)$的乘积，可以将yn=0和yn=1这两种情况写到同一个表达式中，整理后似然值可以写成：
   $$
   L(\theta)=\prod_{i=1}^{n}(h_{\theta}(x^{(i)})^{y^{(i)}}*(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}
   $$
   其中，$x^{(i)}$表示第i个样本。为了把连乘问题简化计算，取对数似然函数，将连乘转化为连加；取负号，将maximize问题转化为minimize问题；并引入平均数操作1/N：
   $$
   J(\theta)=-\frac{1}{N}log(L(\theta))=-\frac{1}{N}\sum_{i=1}^{N}[y^{(i)}logh_{\theta}(x^{(i)})+(1-y^{(i)})log(1-h_{\theta}(x^{(i)}))]
   $$
   上式也被称为交叉熵损失函数

3. 如何选取参数$\boldsymbol{\theta}$使损失函数最小呢，由于损失函数$J(\boldsymbol{\theta})$的Hessian矩阵是正定的，因此$J(\boldsymbol{\theta})$是关于$\boldsymbol{\theta}$的凸函数，为了计算$J(\boldsymbol{\theta})$的最小值时的$\boldsymbol{\theta}$，只需计算$J(\boldsymbol{\theta})$的梯度为0时的$\boldsymbol{\theta}$，即为最优解，下面求$J(\boldsymbol{\theta})$的梯度$\nabla J(\mathbf{\boldsymbol\theta})$
   $$
   \nabla J(\mathbf{\boldsymbol\theta})=(\frac{\partial J(\boldsymbol{\theta}) }{\partial \theta_1},\frac{\partial J(\boldsymbol{\theta}) }{\partial \theta_2},...,\frac{\partial J(\boldsymbol{\theta}) }{\partial \theta_m})
   $$

   $$
   \frac{\partial J(\boldsymbol{\theta}) }{\partial \theta_j}=\frac{\partial }{\partial \theta_j}[-\frac{1}{N}\sum_{i=1}^{N}[y^{(i)}logh_{\theta}(x^{(i)})+(1-y^{(i)})log(1-h_{\theta}(x^{(i)}))]]
   $$

   根据链式求导法则，仔细推导可以得到
   $$
   \frac{\partial J(\boldsymbol{\theta}) }{\partial \theta_j}=\frac{1}{N}\sum_{i=1}^{N}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}
   $$
   要让$\frac{\partial J(\boldsymbol{\theta}) }{\partial \theta_j}=0$，这种情况没有closed-form解，与Linear Regression不同，只能用迭代方法求解。

4. 已知$J(\boldsymbol{\theta})$是凸函数，可以把$J(\boldsymbol{\theta})$看做一个山谷，求$J(\boldsymbol{\theta})$最小的过程可以看做下山的过程，那么根据泰勒Taylor一阶展开，可以得到：
   $$
   J(\boldsymbol{\theta_{t+1}})=J(\boldsymbol{\theta_{t}}+\eta\boldsymbol{\nu } )=J(\boldsymbol{\theta_{t}})+\eta\boldsymbol{\nu }^{T}
   \nabla J(\boldsymbol{\theta_{t}})
   $$
   迭代的目的是让$J(\boldsymbol{\theta_{t+1}})$越来越小，即让$J(\boldsymbol{\theta_{t+1}})< J(\boldsymbol{\theta_{t}})$。η是标量，因为如果两个向量方向相反的话，那么他们的内积最小（为负），也就是说如果方向v与梯度$\nabla J(\boldsymbol{\theta_{t}})$反向的话，那么就能保证每次迭代$J(\boldsymbol{\theta_{t+1}})< J(\boldsymbol{\theta_{t}})$都成立。则我们令下降方向v为：
   $$
   \boldsymbol{\nu} =-\frac{\nabla J(\boldsymbol{\theta_{t}})}{\left \| \nabla J(\boldsymbol{\theta_{t}})\right \|}
   $$
   v是单位向量，v每次都是沿着梯度的反方向走，这种方法称为梯度下降（gradient descent）算法

   即
   $$
   \theta_{j}^{t+1}：=\theta_{j}^{t}-\alpha\frac{1}{N}\sum_{i=1}^{N}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}
   $$
   当达到迭代次数或损失函数很小时，迭代结束，此时的$\boldsymbol{\theta}$即为最优$\boldsymbol{\theta}$

## 关于凸函数的Hessian矩阵正定

### 1阶条件

以一元函数为例，凸函数判定的一阶条件是：

对于定义域内任意两个自变量x1和x2，若函数f满足
$$
f(x_2)\geq f(x_1)+f'(x_1)(x_2-x_1)
$$
则函数f为凸函数

<img src="/wiki/static/images/logistic_regression/tu.png" alt="max_likelihood"/>

如上图所示，直观的理解就是函数曲线始终位于任意一点的切线的上方。

推广到多元函数写为
$$
f(\boldsymbol{x}_2)\geq f(\boldsymbol{x}_1)+\nabla f(\boldsymbol{x}_1)(\boldsymbol{x}_2-\boldsymbol{x}_1)
$$
其中梯度向量为
$$
\nabla f(\mathbf{\boldsymbol x})=\left (\frac{\partial f(\mathbf{\boldsymbol x})  }{\partial x_1},\frac{\partial f(\mathbf{\boldsymbol x})  }{\partial x_2},...,\frac{\partial f(\mathbf{\boldsymbol x})  }{\partial x_n}\right)
$$
也就是对各个变量求偏导构成的向量。

### 2阶条件

直接对多元函数$f(\mathbf{\boldsymbol x})$在$\boldsymbol x_0$处泰勒展开，
$$
f(\boldsymbol{x})= f(\boldsymbol{x}\_0)+\nabla f(\boldsymbol{x}\_0)(\boldsymbol{x}-\boldsymbol{x}\_0)+\frac{1}{2}(\boldsymbol{x}-\boldsymbol{x}\_0)^{T}\boldsymbol H(\boldsymbol{x}\_0)(\boldsymbol{x}-\boldsymbol{x}\_0)
$$
$\boldsymbol H(\boldsymbol{x}\_0)$即$f(\mathbf{\boldsymbol x})$在$\boldsymbol x_0$点的Hessian矩阵，也可以写成$\nabla^2 f(\boldsymbol{x}\_0)$，$\boldsymbol H_{ij}= \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_i \partial x_j}$，写成矩阵形式就是
$$
\begin{bmatrix}
\frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_1^2} & \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_1 \partial x_2}  &\cdots   &  \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_1 \partial x_n}\\\\
 \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_2 \partial x_1} & \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_2^2} & \cdots  & \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_2 \partial x_n} \\\\ 
\vdots  & \vdots  &\ddots  &\vdots  \\\\
  \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_n \partial x_1}& \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_n \partial x_2}  &  \cdots & \frac{\partial^2f(\mathbf{\boldsymbol x}) }{\partial x_n^2}
\end{bmatrix}
$$
可以看出，该矩阵是实对称矩阵。

对于一个凸函数，1st-order condition为$f(\boldsymbol{x})\geq f(\boldsymbol{x}_0)+\nabla f(\boldsymbol{x}_0)(\boldsymbol{x}-\boldsymbol{x}_0)$对任意的$\boldsymbol{x}$和$\boldsymbol{x}_0$都成立，因此式（33）中的$\frac{1}{2}(\boldsymbol{x}-\boldsymbol{x}_0)^{T}\boldsymbol H(\boldsymbol{x}_0)(\boldsymbol{x}-\boldsymbol{x}_0)\geq0$也要对任意的$\boldsymbol{x}$和$\boldsymbol{x}_0$都成立，即$\triangle \boldsymbol{x}^T\boldsymbol H(\boldsymbol{x}_0)\triangle \boldsymbol{x}\geq 0 $对任意$\triangle \boldsymbol{x}$恒成立，而这就是$\boldsymbol H$半正定的充要条件。

那么，要证明交叉熵损失函数的Hessian矩阵半正定，只需要写出它的Hessian矩阵，然后用判定正定矩阵的方法进行判定即可，
$$
\frac{\partial J^2(\boldsymbol{\theta}) }{\partial \theta_j \partial \theta_k}=\frac{1}{N}\sum_{i=1}^{N}(h^2_{\theta}(x^{(i)})\cdot e^{-\theta^Tx^{(i)}}\cdot x_k^{(i)})x_j^{(i)}
$$

## softmax 回归

Softmax回归模型是logistic回归模型在多分类问题上的推广，对于多分类问题，$y^{i}\in  \{1,2,3,...,k \}$，对于给定的测试输入x，我们想用假设函数针对每一个类别$j$估算出概率值$P(y=j|x)$，因此，我们的假设函数将要输出一个 k维的向量（向量元素的和为1）来表示这k个估计的概率值。 具体地说，我们的假设函数$h_{\theta}({x})$形式如下：
$$
h_{\theta}(x^{(i)})=\begin{bmatrix}
P(y^{(i)}=1|x^{(i)};\theta)\\ 
P(y^{(i)}=2|x^{(i)};\theta)\\ 
\vdots \\ 
P(y^{(i)}=k|x^{(i)};\theta)
\end{bmatrix}
=\frac{1}{\sum_{j=1}^{k}e^{\boldsymbol{\theta}{{}}\_j^Tx^{(i)}}}\begin{bmatrix}
e^{\boldsymbol{\theta}\_1^Tx^{(i)}}\\\\
e^{\boldsymbol{\theta}\_2^Tx^{(i)}}\\\\
\vdots \\\\
e^{\boldsymbol{\theta}\_k^Tx^{(i)}}
\end{bmatrix}
$$
其中，$\sum_{j=1}^{k}e^{\boldsymbol{\theta}{{}}\_j^Tx^{(i)}}$这一项是对概率分布做归一化，使得所有类别概率之和为 1 。

用符号$\boldsymbol \theta$来表示所有的模型参数，将$\boldsymbol \theta$用一个$k \times (n+1)$矩阵来表示，n为特征维度数，该矩阵是将$\boldsymbol{\theta}\_1,\boldsymbol{\theta}\_2,...,\boldsymbol{\theta}\_k$按行罗列起来的，如下所示
$$
\boldsymbol \theta=\begin{bmatrix}
\boldsymbol{\theta}\_1^T\\\\ 
\boldsymbol{\theta}\_2^T\\\\
\vdots \\\\
\boldsymbol{\theta}\_k^T
\end{bmatrix}
$$
 softmax 回归算法的代价函数也是logistic回归代价函数的推广，logistic回归代价函数可以改为
<img src="/wiki/static/images/logistic_regression/a.png" alt="noise_data"/>
其中，1{表达式值为真}=1，为示性函数，将logistic回归代价函数推广到softmax的代价函数
$$
\begin{align}
J(\theta)
&=-\frac{1}{N} [\sum_{i=1}^{N}\sum_{j=0}^{k}1 (\{y^{(i)}=j   \})log\frac{e^{\boldsymbol{\theta}{{}}\_j^Tx^{(i)}}}{\sum_{l=1}^{k}e^{\boldsymbol{\theta}{{}}\_l^Tx^{(i)}}}]
\end{align}
$$
Softmax代价函数与logistic 代价函数在形式上非常类似，只是在Softmax损失函数中对类标记的 $k$个可能值进行了累加。注意在Softmax回归中将 $x$分类为类别 $j$的概率为：
$$
p(y^{(i)}=j|x^{(i)};\theta)=\frac{e^{\boldsymbol{\theta}{{}}\_j^Tx^{(i)}}}{\sum_{l=1}^{k}e^{\boldsymbol{\theta}{{}}\_l^Tx^{(i)}}}
$$
对于$J(\theta)$最小化的问题，同样可以用梯度下降法求解。

## 参考

1. [红色石头的机器学习之路](https://redstonewill.github.io/2018/03/17/10/)
2. [机器学习基石课程](https://www.bilibili.com/video/av12463015/?p=41)
3. [softmax](http://deeplearning.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)
4. [怎样理解凸函数与Hessian矩阵半正定](https://www.zhihu.com/question/40181086)