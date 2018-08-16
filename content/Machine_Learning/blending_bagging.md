---
title: "Blending and Bagging"
layout: page
date: 2018-07-12 00:00
---

[TOC]

## Motivation of Aggregation

首先举个例子来说明为什么要使用Aggregation。假如你有T个朋友，每个朋友向你预测推荐明天某支股票会涨还是会跌，对应的建议分别是g1,g2,⋯,gT，那么你该选择哪个朋友的建议呢？即最终选择对股票预测的gt(x)是什么样的？

我们可以有下面4种选择：

1. 第一种方法是从T个朋友中选择一个最受信任，对股票预测能力最强的人，直接听从他的建议就好。这是一种普遍的做法，对应的就是validation思想，即选择犯错误最小的模型。
2. 第二种方法，同时考虑T个朋友的建议，将所有结果做个投票，一人一票，最终决定出对该支股票的预测。这种方法对应的是uniformly思想。
3. 第三种方法，如果每个朋友水平不一，有的比较厉害，投票比重应该更大一些，有的比较差，投票比重应该更小一些。那么，仍然让T个朋友进行投票，只是每个人的投票权重不同。这种方法对应的是non-uniformly的思想。
4. 第四种方法与第三种方法类似，但是权重不是固定的，根据不同的条件，给予不同的权重。比如如果是传统行业的股票，那么给这方面比较厉害的朋友较高的投票权重，如果是服务行业，那么就给这方面比较厉害的朋友较高的投票权重。

Aggregation的思想与这个例子是类似的，即把多个hypothesis结合起来，得到更好的预测效果。

 将刚刚举的例子的各种方法用数学化的语言和机器学习符号归纳表示出来，其中$G(x)$表示最终选择的模型。

第一种方法对应的模型： 
$$
G(x)=g_{t_{\*}}(x) \     \    with \ t_*  =\ \mathop{\arg\min}\_{t \in 1,2...,T}E_{val}(g_{t}^{-})
$$
注意这里是通过验证集来选择最佳模型，不能使用$E_{in}(g_{t}^{})$来代替$E_{val}(g_{t}^{-})$。经过Validation，选择最小的$E_{val}(g_{t}^{-})$，保证$E_{out}$最小，从而将对应的模型作为最佳的选择

第二种方法对应的模型： 
$$
G(x)=sign(\sum_{t=1}^{T}1 \cdot g_t(x))
$$
第三种方法对应的模型
$$
G(x)=sign(\sum_{t=1}^{T}\alpha_t\cdot g_t(x)),\alpha_t\geqslant 0
$$
第四种方法对应的模型
$$
G(x)=sign(\sum_{t=1}^{T}q_t(x)\cdot g_t(x)),q_t(x)\geqslant 0
$$
但是第一种方法只是从众多可能的hypothesis中选择最好的模型，并不能发挥集体的智慧。而Aggregation的思想是博采众长，将可能的hypothesis优势集合起来，将集体智慧融合起来，使预测模型达到更好的效果。

下面先来看一个例子，通过这个例子说明为什么Aggregation能work得更好。

<img src="/wiki/static/images/adaboost/b1.png" alt="joey"/>

如上图所示，平面上分布着一些待分类的点。如果要求只能用一条水平的线或者垂直的线进行分类，那不论怎么选取直线，都达不到最佳的分类效果。这实际上就是上面介绍的第一种方法：validation。但是，如果可以使用集体智慧，比如一条水平线和两条垂直线组合而成的图中折线形式，就可以将所有的点完全分开，得到了最优化的预测模型。

这个例子表明，通过将不同的hypotheses均匀地结合起来，得到了比单一hypothesis更好的预测模型。这就是aggregation的优势所在，它提高了预测模型的power，起到了**特征转换（feature transform）**的效果。

我们再从另外一方面来看，同样是平面上分布着一些待分类的点，使用PLA算法，可以得到很多满足条件的分类线，如下图所示：

<img src="/wiki/static/images/adaboost/b2.png" alt="joey"/>

这无数条PLA选择出来的灰色直线对应的hypothesis都是满足分类要求的。但是我们最想得到的分类直线是中间那条距离所有点都比较远的黑色直线，这与之前SVM目标是一致的。如果我们将所有可能的hypothesis结合起来，以投票的方式进行组合选择，最终会发现投票得到的分类线就是中间和黑色那条。这从哲学的角度来说，就是对各种效果较好的可能性进行组合，得到的结果一般是中庸的、最合适的，即对应图中那条黑色直线。所以，aggregation也起到了正则化（regularization）的效果，让预测模型更具有代表性。

基于以上的两个例子，我们得到了aggregation的两个优势：feature transform和regularization。我们之前在机器学习基石课程中就介绍过，feature transform和regularization是对立的，还把它们分别比作踩油门和踩刹车。如果进行feature transform，那么regularization的效果通常很差，反之亦然。也就是说，单一模型通常只能倾向于feature transform和regularization之一，在两者之间做个权衡。但是aggregation却能将feature transform和regularization各自的优势结合起来，好比把油门和刹车都控制得很好，从而得到不错的预测模型。

## Uniform Blending

那对于我们已经选择的性能较好的一些$g_t$，如何将它们进行整合、合并，来得到最佳的预测模型呢？这个过程称为blending。

### 分类

最常用的一种方法是uniform blending，应用于classification分类问题，做法是将每一个可能的矩赋予权重1，进行投票，得到的$G(x)$表示为：
$$
G(x)=sign(\sum_{t=1}^{T}1 \cdot g_t(x))
$$
这种方法对应三种情况：

1. 第一种情况是每个候选的$g_t$都完全一样，这跟选其中任意一个$g_t$效果相同；
2. 第二种情况是每个候选的$g_t$都有一些差别，这是最常遇到的，大都可以通过投票的形式使多数意见修正少数意见，从而得到很好的模型；
3. 第三种情况是多分类问题，选择投票数最多的那一类即可。

### 回归

如果是regression回归问题，uniform blending的做法很简单，就是将所有的$g_t$求平均值
$$
G(x)=\frac{1}{T}\sum_{t=1}^{T}g_t(x)
$$
uniform blending for regression对应两种情况：

1. 第一种情况是每个候选的$g_t$都完全一样，这跟选其中任意一个$g_t$效果相同；
2. 第二种情况是每个候选的$g_t$都有一些差别，有的$g_t>f(x)$，有的$g_t<f(x)$，此时求平均值的操作可能会消去这种大于和小于的影响，从而得到更好的回归模型。因此，从直觉上来说，求平均值的操作更加稳定，更加准确

对于uniform blending，一般要求每个候选的$g_t$都有一些差别。这样，通过不同$g_t$的组合和集体智慧，都能得到比单一$g_t$更好的模型。

刚才我们提到了uniform blending for regression中，计算$g_t$的平均值可能比单一的$g_t$更稳定，更准确。下面进行简单的推导和证明，对于某一个单一的样本点$\boldsymbol{x}$，以平方误差为例，
$$
\begin{align*}
avg((g_t(\boldsymbol{x})-f(\boldsymbol{x}))^2)  &=avg (g_t^2-2g_tf+f^2)\\\\
 &=avg (g_t^2)-2Gf+f^2\\\\
 &=avg (g_t^2)-G^2+(G-f)^2\\\\
 &=avg (g_t^2)-2G^2+G^2+(G-f)^2\\\\
 &=avg (g_t^2)-2g_tG^2+G^2+(G-f)^2\\\\
 &=avg (g_t-G)^2+(G-f)^2\\\\
\end{align*}
$$
推导过程中注意$G(t)=avg(g_t)$，即对所有的$g_t$做$avg$

刚才是对单一的$\boldsymbol{x}$进行证明，如果从期望角度，对整个$\boldsymbol{x}$分布进行上述公式的整理，得到：
$$
\begin{align*}
avg(E_{out}(g_t))&=avg(\boldsymbol{\varepsilon}  (g_t-G)^2)+E_{out}(G)\\\\
& \geqslant E_{out}(G)
\end{align*}
$$
也就是说，$G$的表现要比$g_t$的平均表现要好，**注意，是$g_t$的平均表现，而不是某一个$g_t$的表现，视频中是这样说的，“最好的$g_t$的表现是否比G好，我们不知道，但是所有的$g_t$的平均表现要比G差“**，这个结果给了我们做aggregation这件事一点点的保证，那么怎样运用这个事情呢？现在假设一个抽象的机器学习过程：

------

for t=1,2...,T

1. 从$P^N$这个分布（i.i.d）产生包含N个样本点的数据集$D_t$，每一轮都有新的N个样本点
2. 用演算法$\mathcal{A}$从数据集$D_t$中学习得到$g_t$

对$g_t$求平均得到$G$，当做无限多次，即T趋向于无穷大的时候,令
$$
\bar g=\lim_{T \to \infty }=\lim_{T \to \infty }\frac{1}{T}\sum_{t=1}^{T}g_t=\boldsymbol{\varepsilon}_D\mathcal{A}(D)
$$
 $\bar g$表示对产生$D_t$资料的过程的平均

------

现在，我们把刚才的结果中的$G$推广到无限大，即用$\bar g$来代替$G$
$$
avg(E_{out}(g_t))=avg(\boldsymbol{\varepsilon}  (g_t-\bar g)^2)+E_{out}(\bar g)
$$
这个式子的物理意义是什么呢？

等式左边$avg(E_{out}(g_t))$可以理解为演算法$\mathcal{A}$的期望表现，（**注意，我的理解是，这里的演算法$\mathcal{A}$指的并不是数值优化算法，而是统计学习方法那本书中所提到的机器学习三要素：模型+策略+算法**），即$\mathcal{A}$看了各式各样不同的

$D_t$后，所产生的各个$g_t$的期望表现的平均。

等式右边$E_{out}(\bar g)$可以理解为所有$g_t$的共识(consensus)$\bar g$的表现，称为偏差bias，即共识$\bar g$跟我们想要的目标函数$f$差多远。

另外一项$avg(\boldsymbol{\varepsilon}  (g_t-\bar g)^2$可以理解为$g_t$和共识$\bar g$到底差多远，称为方差variance，即$g_t$的意见有多不一样

于是，我们就把演算法$\mathcal{A}$的期望表现拆成了bias+variance两项，一个是所有$g_t$的共识，一个是不同$g_t$之间的差距是多少。

而uniform blending的操作时求平均的过程，会使大家的意见趋于一致，这样就削减弱化了上式第一项variance的值，从而演算法的表现就更好了，能得到更加稳定的表现。

## Linear and any blending

### linear blending

上一部分讲的是uniform blending，即每个$g_t$所占的权重都是1，求平均的思想。下面我们将介绍linear blending，每个$g_t$赋予的权重$\alpha_t$并不相同，其中$\alpha_t \geq 0$。我们最终得到的预测结果等于所有$g_t$的线性组合。


$$
G(x)=sign(\sum_{t=1}^{T}\alpha_t\cdot g_t(x)),\alpha_t\geqslant 0
$$
如何确定$α\_t$的值，方法是利用误差最小化的思想，找出最佳的$α\_t$，使$E_{in}(α)$取最小值。例如对于linear blending for regression，$E_{in}(α)$可以写成
$$
\mathop{\min}\_{\alpha\geq 0} \frac{1}{N}\sum_{n=1}^{N}(y_n-\sum_{t=1}^{T}\alpha\_tg_t(\boldsymbol x_n))^2
$$
下图左边形式，其中$α\_t$是带求解参数，$g_t(\boldsymbol x_n)$是每个g得到的预测值。这种形式很类似于经过特征转换的linear regression模型
$$
\mathop{\min}\_{\boldsymbol w_i} \frac{1}{N}\sum_{n=1}^{N}(y_n-\sum_{t=1}^{T}\boldsymbol w_i\phi _i(\boldsymbol x_n))^2
$$
唯一不同的就是linear blending for regression中$α_t≥0$，而linear regression中${\boldsymbol w_i}$没有限制。

 $α_t≥0$这个条件是否一定要成立呢？如果$α_t<0$，会带来什么后果呢？其实$α_t<0$并不会影响分类效果，只需要将正类看成负类，负类当成正类即可。例如分类问题，判断该点是正类对应的$α_t<0$，则它就表示该点是负类，且对应的$-α_t>0$。如果我们说这个样本是正类的概率是-99%，意思也就是说该样本是负类的概率是99%。$α_t≥0$和$α_t<0$的效果是一致的。所以，我们可以把$α_t≥0$这个条件舍去，这样linear blending就可以使用常规方法求解。

在实际中，不同的$g_t$通常是来自不同的model通过各自求最小化$E_{in}$得到的，即
$$
g_1\in\mathcal{H_1},g_2\in\mathcal{H}\_2,...,g_T\in\mathcal{H}\_T\  \\\
by\ minimum\ E_{in}
$$
那么如果我们通过最小化$E_{in}$来从$g_1,g_2,...,g_T$中选择一个表现最好的，我们会付出$d_{vc}(\bigcup_{t=1}^{T}\mathcal{H}\_t)$的模型复杂度的代价，这个代价很大；如果我们将$g_1,g_2,...,g_T$做linear blending，并通过最小化$E\_{in}$来确定系数$α\_t$的话，那么我们付出的模型复杂度代价比$d_{vc}(\bigcup\_{t=1}^{T}\mathcal{H}\_t)$还要大，因此，在模型选择的时候要用单独的验证集validation上的表现$E_{val}$来代替$E_{in}$，具体的做法是：

------

将数据集$D$分为训练集$D_{trian}$和验证集$D_{validation}$

从$D_{trian}$中得到$g_1^-,g_2^-,...,g_T^-$，有了这些$g^-$之后，通过这些$g^-$将$D_{validation}$中的数据点$({\boldsymbol x_n},y_n)$转化到z空间中$({\boldsymbol z_n}=\boldsymbol\phi^-(\boldsymbol x_n),y_n)$，其中$\boldsymbol\phi^-(\boldsymbol x)=(g_1^-(\boldsymbol x),g_2^-(\boldsymbol x),...,g_T^-(\boldsymbol x))$

在这些新的点$({\boldsymbol z_n},y_n)$上做线性模型的学习，来确定linear blending的系数$α_t$

最后返回$G(\boldsymbol x)=LinH(innerprod(\boldsymbol\alpha,\boldsymbol\phi(\boldsymbol x)))$，其中$\boldsymbol\phi(\boldsymbol x)=g_{1}{(\boldsymbol x)},g_{2}{(\boldsymbol x)},...,g_{T}{(\boldsymbol x)})$，**这里关于$g_{t}{(\boldsymbol x)}$和$g_{t}^-{(\boldsymbol x)}$的区别，我的理解是，假设$g_{t}^-$是从训练集$D_{trian}$中学到的，$g_{t}^-{(\boldsymbol x)}$指的是假设$g_{t}^-$在训练集$D_{trian}$上的预测，而$g_{t}{(\boldsymbol x)}$指的是假设$g_{t}^-$在整个数据集$D$上的预测**

------



### any blending

除了linear blending之外，还可以使用任意形式的blending。linear blending中，$G(t)$是$g(t)$的线性组合；any blending中，$G(t)$可以是$g(t)$的任何函数形式（非线性）。这种形式的blending也叫做Stacking。any blending的优点是模型复杂度提高，更容易获得更好的预测模型；缺点是复杂模型也容易带来过拟合的危险。所以，在使用any blending的过程中要时刻注意避免过拟合发生，通过采用regularization的方法，让模型具有更好的泛化能力。

## Bagging(Bootstrap Aggregation)

总结一些上面讲的内容，blending的做法就是将已经得到的矩$g_t$进行aggregate的操作，具体的aggregation形式包括：uniform，non-uniforn和conditional。 

现在考虑一个问题：如何得到不同的$g_t$呢？可以选取不同模型$\mathcal{H}$；可以设置不同的参数，例如$η$、迭代次数n等；可以由算法的随机性得到，例如PLA、随机种子等；可以选择不同的数据样本等。这些方法都可能得到不同的$g_t$。

那如何利用已有的一份数据集来构造出不同的$g_t$呢？回顾一下之前介绍的bias-variance，即一个演算法的平均表现可以被拆成两项，一个是所有$g_t$的共识（bias），一个是不同$g_t$之间的差距是多少（variance）。**其中每个$g_t$都是从新的数据集$D_{t}$得来的，一共有无穷多个$g_t$**，那么我们在只有一份数据集的情况下，为了模拟这一过程，需要做两个妥协：

- 有限的T
- 由已有的数据集D构造出$D_{t}$，独立同分布

第一个条件没有问题，第二个近似条件的做法就是bootstrapping。

bootstrapping的做法是，假设有N笔资料，先从中选出一个样本，再放回去，再选择一个样本，再放回去，共重复N次。这样我们就得到了一个新的N笔资料，这个新的$D_{t}$中可能包含原D里的重复样本点，也可能没有原D里的某些样本，$D_{t}$与D类似但又不完全相同。值得一提的是，抽取-放回的操作不一定非要是N，次数可以任意设定。例如原始样本有10000个，我们可以抽取-放回3000次，得到包含3000个样本的$D_{t}$也是完全可以的。利用bootstrap进行aggragation的操作就被称为bagging。

下面举个实际中Bagging Pocket算法的例子。如下图所示，先通过bootstrapping得到25个不同样本集，再使用pocket算法得到25个不同的$g_t$，每个pocket算法迭代1000次。最后，再利用blending，将所有的$g_t$融合起来，得到最终的分类线，如图中黑线所示。可以看出，虽然bootstrapping会得到差别很大的分类线（灰线），但是经过blending后，得到的分类线效果是不错的，则bagging通常能得到最佳的分类模型。

<img src="/wiki/static/images/adaboost/b3.png" alt="joey"/>

值得注意的是，只有当演算法对数据样本分布比较敏感的情况下，才有比较好的表现。

## 总结

本节课主要介绍了blending和bagging的方法，它们都属于aggregation，即将不同的$g_t$合并起来，利用集体的智慧得到更加优化的$G(t)$。Blending通常分为三种情况：Uniform Blending，Linear Blending和Any Blending。其中，uniform blending采样最简单的“一人一票”的方法，linear blending和any blending都采用标准的two-level learning方法，类似于特征转换的操作，来得到不同$g_t$的线性组合或非线性组合。最后，我们介绍了如何利用bagging（bootstrap aggregation），从已有数据集D中模拟出其他类似的样本$D_{t}$，而得到不同的$g_t$，再合并起来，优化预测模型。

## 参考

[红色石头机器学习之路](https://blog.csdn.net/red_stone1/article/details/74937526)

[林轩田 机器学习技法](https://www.bilibili.com/video/av12469267/?p=28)



