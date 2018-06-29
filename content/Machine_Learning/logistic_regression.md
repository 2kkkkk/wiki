---
title: "Logistic Regression"
layout: page
date: 2018-06-29 00:00
---

[TOC]

##Logistic Regression Problem

一个心脏病预测的问题：根据患者的年龄、血压、体重等信息，来预测患者是否会有心脏病。很明显这是一个二分类问题，其输出y只有{-1,1}两种情况。
但是，如果我们想知道的不是患者有没有心脏病，而是到底患者有多大的几率是心脏病。这表示，我们更关心的是目标函数的值（分布在0,1之间），表示是正类的概率（正类表示是心脏病）。这跟我们原来讨论的二分类问题不太一样，我们把这个问题称为软性二分类问题（’soft’ binary classification）。这个值越接近1，表示正类的可能性越大；越接近0，表示负类的可能性越大。
目标函数target function: $$f(x)=P(+1|x)\in [0,1]$$
对于软性二分类问题，理想的label为概率值，是分布在[0,1]之间的具体值，但是实际数据中label是0或者1，我们可以把实际中的数据看成是理想数据加上了噪声的影响。
\begin{matrix}
 ideal(noiseless)data \\ 
 \left \{ (\mathbf{x}_{1},y_{1}=0.9=P(+1|\mathbf{x}_{1}) \right \}\\
 \left \{ (\mathbf{x}_{2},y_{2}=0.2=P(+1|\mathbf{x}_{2}) \right \}  \\
 .\\
.\\
.\\
\left \{ (\mathbf{x}_{N},y_{N}=0.3=P(+1|\mathbf{x}_{N}) \right \}
\end{matrix}

\begin{matrix}
 actual(noisy)data \\ 
 \left \{ (\mathbf{x}_{1},y_{1}=1\sim P(y|\mathbf{x}_{1}) \right \}\\
 \left \{ (\mathbf{x}_{2},y_{2}=-1\sim P(y|\mathbf{x}_{2}) \right \}  \\
 .\\
.\\
.\\
\left \{ (\mathbf{x}_{N},y_{N}=+1 \sim P(y|\mathbf{x}_{N}) \right \}
\end{matrix}
和hard binary classfication相比，使用相同的数据集，但学到不同的target function。

如果目标函数是$(f(x)=P(+1|x)∈[0,1])$的话，我们如何找到一个好的Hypothesis跟这个目标函数很接近呢？
首先，根据我们之前的做法，对所有的特征值进行加权处理，计算的结果s，我们称之为 risk score：
$$for\  \mathbf{x}=(x_{0},x_{1},x_{2},\cdots ,x_{d})\\s=\sum_{i=0}^{d}w_{i}x_{i}$$
但是特征加权和s∈(−∞,+∞)，如何将s值限定在[0,1]之间呢？一个方法是使用sigmoid Function，记为θ(s)
<img src="/wiki/static/images/logistic_regression/sigmoid.png" alt="sigmoid function"/>

Sigmoid Function函数记为$(\theta (s)=\frac{1}{1+e^{-s}})$，满足$(\theta (-\infty)=0,\theta (0)=0.5,\theta (+\infty)=1)$。这个函数是平滑的、单调的S型函数。则对于逻辑回归问题，hypothesis就是这样的形式：
$$h(\mathbf{x})=\frac{1}{1+e^{-\mathbf{w}^{T}\mathbf{x}}}$$

那我们的目标就是求出这个预测函数h(x)，使它接近目标函数f(x)

##Logistic Regression Error
现在我们将Logistic Regression与之前讲的Linear Classification、Linear Regression做个比较：

<img src="/wiki/static/images/logistic_regression/lr_compare.png" alt="lr_compare"/>

这三个线性模型都会用到线性scoring function $( s=\mathbf{w}^{T}\mathbf{x})$。linear classification的误差使用的是0/1 error，linear regression的误差使用的是squared error，那么logistic regression的误差该如何定义呢？
先介绍一下“似然性”的概念。目标函数$(f(x)=P(+1|x))$，如果我们找到了hypothesis很接近target function， 也就是说，在所有的Hypothesis集合中找到一个hypothesis与target function最接近，那么由该hypothesis生成数据集D的似然值应该与由target function 生成数据集D的概率值很接近。

<img src="/wiki/static/images/logistic_regression/max_likelihood.png" alt="max_likelihood"/>

logistic function: $ (h(\mathbf{x})=\theta (\mathbf{w}^{T}\mathbf{x}))$ 满足一个性质：1−h(x)=h(−x)， 因此似然函数可以写成：
$$likelihood(h)=P(\mathbf{x}_{1})h(+\mathbf{x}_{1})\times P(\mathbf{x}_{2})h(-\mathbf{x}_{2})\times \cdots \times P(\mathbf{x}_{N})h(+\mathbf{x}_{N})$$
因为$(P(\mathbf{x}_{n}))$
对所有的hypothesis来说，都是一样的，所以我们可以忽略它，那么可以得到$(likelihood(h))$正比于所有$(h(y_{n}\mathbf{x}_{n}))$的乘积，通常情况下target function $(f)$ 生成数据集D的probability很大，因此我们的目标就是让所有$(h(y_{n}\mathbf{x}_{n}))$的乘积值最大化。
$$\max_{h}\ likelihood(logistic \ h)\propto \prod_{n=1}^{N} h(y_{n}\mathbf{x}_{n})$$
将$( h(\mathbf{x})=\theta (\mathbf{w}^{T}\mathbf{x}))$代入
$$\max_{\mathbf{w}}\ likelihood(logistic \ h)\propto \prod_{n=1}^{N} \theta (y_{n}\mathbf{w^{T}}\mathbf{x}_{n})$$
为了把连乘问题简化计算，我们可以引入ln操作，让连乘转化为连加
$$\max_{\mathbf{w}}ln \prod_{n=1}^{N} \theta (y_{n}\mathbf{w^{T}}\mathbf{x}_{n})=\max_{\mathbf{w}}\sum_{n=1}^{N}ln\theta (y_{n}\mathbf{w^{T}}\mathbf{x}_{n})$$
接着，我们将maximize问题转化为minimize问题，并引入平均数操作1/N：
$$\min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}-ln\theta (y_{n}\mathbf{w^{T}}\mathbf{x}_{n})$$
将logistic function的表达式$(\theta (s)=\frac{1}{1+e^{-s}})$代入得到
$$\min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}}) \Rightarrow \min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}err(\mathbf{w},\mathbf{x_{n}},y_{n})$$
至此，我们得到了logistic regression的err function，称之为cross-entropy error交叉熵损失函数。
##Gradient of Logistic Regression Error
我们已经推导了$(E_{in})$的表达式，那接下来的问题就是如何找到合适的向量$(\mathbf{w})$，让$(E_{in})$最小。
$$\min_{\mathbf{w}}E_{in}(\mathbf{w})=\min_{\mathbf{w}}\frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}})$$
Logistic Regression的$(E_{in})$是关于$(\mathbf{w})$的连续、可微、二次可微的凸曲线（开口向上），根据之前Linear Regression的思路，我们只要计算$(E_{in})$的梯度为零时的$(\mathbf{w})$，即为最优解。

<img src="/wiki/static/images/logistic_regression/lr_Ein.png" alt="lr_Ein"/>

根据链式求导法则，计算$(E_{in})$梯度
$$\frac{\partial E_{in}(\mathbf{w})}{\partial w_{i}}=\frac{\partial \frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}})}{\partial w_{i}}\\= \frac{1}{N}\sum_{n=1}^{N}\frac{\partial ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}})}{\partial w_{i}} \\=     \frac{1}{N}\sum_{n=1}^{N} \frac{1}{1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}}}\frac{\partial (1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}})}{\partial w_{i}}    \\=    \frac{1}{N}\sum_{n=1}^{N}(-y_{n}x_{n,i}) \frac{1}{1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}}} e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}}\\= \frac{1}{N}\sum_{n=1}^{N}(-y_{n}x_{n,i})\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}_{n})
$$
写成向量的形式
$$
\nabla E_{in}(\mathbf{w})=\frac{1}{N}\sum_{n=1}^{N}\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}_{n})(-y_{n}\mathbf{x_{n}})
$$
为了计算$(E_{in}(\mathbf{w}))$最小值，就要找到让$(
\nabla E_{in}(\mathbf{w}))$等于0的位置
上式中，$(
\nabla E_{in}(\mathbf{w}))$可以看成是对$((-y_{n}\mathbf{x_{n}}))$的加权平均，权重为$(\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}))$，要想使加权和为0，一种情况是所有的权重$(\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}))$都是0，那么可以保证$(
\nabla E_{in}(\mathbf{w}))$为0，$(\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}))$是sigmoid function，根据其特性，只要让$(-y_{n}\mathbf{w^{T}}\mathbf{x}_{n})$≪0，即$(y_{n}\mathbf{w^{T}}\mathbf{x}_{n})$≫0。$(y_{n}\mathbf{w^{T}}\mathbf{x}_{n})$≫0表示对于所有的点，$(y_{n}和\mathbf{w^{T}}\mathbf{x}_{n})$都是同号的，这表示数据集D必须是全部线性可分的才能成立。
但是实际情况不可能保证数据集线性可分，在非线性可分的情况下，只能通过使加权和为零，来求解w。这种情况没有closed-form解，与Linear Regression不同，只能用迭代方法求解。
先来回顾一下PLA算法，对误分类点，更新权重$(\mathbf{w})$，对于正确分类点，则不更新，PLA的迭代优化过程表示如下：
$$\mathbf{w}_{t+1}=\mathbf{w}_{t}+ \left [sign(\mathbf{w}_{t}^{T})\neq y_{n}   \right ]y_{n}\mathbf{x}_{n}\\=\mathbf{w}_{t}+ \underbrace{1}\cdot \underbrace{\left [sign(\mathbf{w}_{t}^{T})\neq y_{n}   \right ]y_{n}\mathbf{x}_{n}} $$
上式可以这么理解，w每次更新包含两个内容：一个是每次更新的方向$(y_{n}\mathbf{x}_{n})$，用$(\mathbf{\nu })$表示，另一个是每次更新的步长$(\mathbf{\eta })$。参数($(\mathbf{\nu }$,$\mathbf{\eta })$)和终止条件决定了我们的迭代优化算法。

##Gradient Descent
迭代优化让每次w都有更新：
$$for\ t=0,1,2\cdots \\\mathbf{w}_{t+1}=\mathbf{w}_{t}+\mathbf{\eta }\mathbf{\nu }$$
我们把$(E_{in})$曲线看做是一个山谷的话，要求$(E_{in})$最小，即可比作下山的过程。整个下山过程由两个因素影响：一个是下山的单位方向v；另外一个是下山的步长η。
<img src="/wiki/static/images/logistic_regression/downhill.png" alt="downhill"/>

利用微分思想和线性近似，假设每次下山我们只前进一小步，即η很小，那么根据泰勒Taylor一阶展开，可以得到：
$$E_{in}(\mathbf{w}_{t+1})=E_{in}(\mathbf{w}_{t}+\eta\boldsymbol{\nu } )=E_{in}(\mathbf{w}_{t})+\eta\boldsymbol{\nu }^{T}
\nabla E_{in}(\mathbf{w}_{t})$$
迭代的目的是让$(E_{in})$越来越小，即让$(E_{in}(\mathbf{w}_{t+1}) < E_{in}(\mathbf{w}_{t}))$。η是标量，因为如果两个向量方向相反的话，那么他们的内积最小（为负），也就是说如果方向v与梯度$(\nabla E_{in}(\mathbf{w}_{t}))$反向的话，那么就能保证每次迭代$(E_{in}(\mathbf{w}_{t+1}) < E_{in}(\mathbf{w}_{t}))$都成立。则我们令下降方向v为：
$$\boldsymbol{\nu} =-\frac{\nabla E_{in}(\mathbf{w}_{t})}{\left \| \nabla E_{in}(\mathbf{w}_{t}) \right \|}$$
v是单位向量，v每次都是沿着梯度的反方向走，这种方法称为梯度下降（gradient descent）算法。那么每次迭代公式就可以写成：
$$\mathbf{w}_{t+1}=\mathbf{w}_{t}+-\eta \frac{\nabla E_{in}(\mathbf{w}_{t})}{\left \| \nabla E_{in}(\mathbf{w}_{t}) \right \|}$$
下面讨论一下$\eta$的大小对迭代优化的影响：$(\eta)$如果太小的话，那么下降的速度就会很慢；$(\eta$)如果太大的话，那么之前利用Taylor展开的方法就不准了，造成下降很不稳定，甚至会上升。因此，$(\eta)$应该选择合适的值，一种方法是在梯度较小的时候，选择小的$(\eta)$，梯度较大的时候，选择大的$(\eta)$，即$(\eta)$正比于$(\left \| \nabla E_{in}(\mathbf{w}_{t}) \right \|)$。这样保证了能够快速、稳定地得到最小值$(E_{in}(\mathbf{w}))$。
<img src="/wiki/static/images/logistic_regression/eta.png" alt="eta"/>

对学习速率$(\eta)$做个更修正，梯度下降算法的迭代公式可以写成：

$$\mathbf{w}_{t+1}=\mathbf{w}_{t}+-{\eta}'\nabla E_{in}(\mathbf{w}_{t})$$

其中：
$${\eta}'=\frac{-\eta }{\left \| \nabla E_{in}(\mathbf{w}_{t}) \right \|}$$
总结一下基于梯度下降的Logistic Regression算法步骤如下：

 1. 初始化w0 
 2. 计算梯度$(\nabla E_{in}(\mathbf{w})=\frac{1}{N}\sum_{n=1}^{N}\theta (-y_{n}\mathbf{w^{T}}\mathbf{x}_{n})(-y_{n}\mathbf{x_{n}})
)$
 3. 迭代更新$(\mathbf{w}_{t+1}=\mathbf{w}_{t}+-{\eta}'\nabla E_{in}(\mathbf{w}_{t}))$
 4. 满足$(\nabla E_{in}(\mathbf{w}_{t}))$≈0或者达到迭代次数，迭代结束

##总结
首先，从逻辑回归的问题出发，将P(+1|x)作为目标函数，将θ(wTx)作为hypothesis。接着，根据极大似然准则定义了logistic regression的err function，称之为cross-entropy error。然后，我们计算logistic regression error的梯度，最后，通过梯度下降算法，计算最优$(\mathbf{w}_{t})$

## 问题
Q：如何证明交叉熵损失函数$(\frac{1}{N}\sum_{n=1}^{N}ln(1+e^{-y_{n}\mathbf{w^{T}}\mathbf{x}_{n}}))$是关于$\mathbf{w}$的的凸函数？
A：机器学习基石视频里给出了思路：求出二次微分的矩阵，该矩阵是正定的，则说明函数是convex的。
**需要自己证明一下！！！**

##参考

 1. [红色石头的机器学习之路](https://redstonewill.github.io/2018/03/17/10/)
 2. [机器学习基石课程](https://www.bilibili.com/video/av12463015/?p=41)