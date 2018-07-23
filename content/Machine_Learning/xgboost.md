---
title: "Xgboost"
layout: page
date: 2018-07-17 00:00
---

[TOC]

## xgb

xgb本质上仍然是梯度提升树。xgb对应的模型就是一堆cart树，也就是一堆的if else。

xgboost为什么使用CART树而不是用普通的决策树呢？

简单讲，对于分类问题，由于CART树的叶子节点对应的值是一个实际的分数，而非一个确定的类别，这将有利于实现高效的优化算法。xgboost出名的原因一是准，二是快，之所以快，其中就有选用CART树的一份功劳。

### 模型

xgboost模型的数学表示为：
$$
\hat{y_i}=\sum_{m=1}^{M}f_m(x_i),f_m\in\mathcal{F}
$$
这里的M就是树的棵数，$\mathcal{F}$表示所有可能的CART树，$f_m$表示一棵具体的CART树。这个模型由M棵CART树组成，模型的参数就是这M棵CART树的参数。

### 目标函数

采用结构风险最小化的策略，模型的目标函数为
$$
obj(\theta)=\sum_{i=1}^{n}l(y_i,\hat{y_i})+\sum_{m=1}^{M}\Omega (f_m)
$$
这个目标函数包含两部分，第一部分就是损失函数，第二部分就是正则项，这里的正则化项由K棵树的正则化项相加而来。

### 前向分步算法

由式（2）可以看到，xgb是一个加法模型，根据前向分步算法，我们不直接优化整个目标函数，而是分步骤优化目标函数，首先优化第一棵树，完了之后再优化第二棵树，直至优化完M棵树

第t步时，在现有的t-1棵树的基础上，找到使得目标函数最小的那棵CART树，即为第t棵树
$$
\begin{align*}
obj^{(t)}&=\sum_{i=1}^{n}l(y_i,\hat{y_i}^{(t)})+\sum_{m=1}^{t}\Omega (f_m)\\
&=\sum_{i=1}^{n}l(y_i,\hat{y_i}^{(t-1)}+f_t(x_i))+\Omega (f_t)+\sum_{m=1}^{t-1}\Omega (f_m)\\
&=\sum_{i=1}^{n}l(y_i,\hat{y_i}^{(t-1)}+f_t(x_i))+\Omega (f_t)+constant
\end{align*}
$$
上式中的constant就是前t-1棵树的复杂度

借鉴GBDT的梯度提升思想，对于一般的损失函数$l(y_i, \hat{y_i})$来说，我们对$l(y_i, \hat{y_i})$在当前模型$f(x)=\sum_{m=1}^{t-1}f_m(x)$处做二阶泰勒展开，
$$
obj^{(t)}=\sum_{i=1}^{n}[l(y_i,\hat{y_i}^{(t-1)})+g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)]+\Omega (f_t)+constant
$$
其中，
$$
\begin{align*}
g_i=\frac{\partial l(y_i,\hat{y_i}^{(t-1)})}{\partial \hat{y_i}^{(t-1)}}\\
h_i=\frac{\partial^2 l(y_i,\hat{y_i}^{(t-1)})}{\partial (\hat{y_i}^{(t-1)})^2}\\
\end{align*}
$$
分别表示损失函数在当前模型的一阶导和二阶导，每一个样本都可以计算出该样本点的$g_i$和$h_i$，而且样本点之间的计算可以独立进行，互不影响，也就是说，可以并行计算。

对式（3）进行化简，去掉与$f_t(x)$无关的项，得到
$$
obj^{(t)}=\sum_{i=1}^{n}[g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)]+\Omega (f_t)
$$
式（4）就是每一步要优化的目标函数。

### 重写优化函数

每一步学到的CART树可以表示成
$$
f_m(x)=w_{q(x)},w\in R^T,q:R^d\rightarrow  \left \{ 1,2,...,T \right \}
$$
其中T为叶子节点个数，q(x)是一个映射，用来将样本映射成1到T的某个值，也就是把它分到某个叶子节点，q(x)其实就代表了CART树的结构。$w_q(x)$自然就是这棵树对样本x的预测值了。

因此，树的复杂度可以表示为：
$$
\Omega(f)=

\gamma T+\frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2
$$
$\gamma $和$\lambda$ 越大，树越简单。

为什么xgboost要选择这样的正则化项？很简单，好使！效果好才是真的好。

将复杂度代入式（4）并做变形，得到
$$
\begin{align*}
obj^{(t)}&=\sum_{i=1}^{n}[g_iw_{q(x_i)}+\frac{1}{2}h_iw^2_{q(x_i)}]+\gamma T+\frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2\\
&=\sum_{j=1}^{T}[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)w_j^2]+\gamma T
\end{align*}
$$
$I_j$代表一个集合，集合中每个值代表一个训练样本的序号，整个集合就是被第t棵CART树分到了第j个叶子节点上的所有训练样本。令$G_j=\sum_{i\in I_j}g_i$和$H_j=\sum_{i\in I_j}h_i$
$$
obj^{(t)}=\sum_{j=1}^{T}[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2]+\gamma T
$$
对于第t棵CART树的某一个确定的结构（可用q(x)表示），所有的Gj和Hj都是确定的。而且上式中各个叶子节点的值wj之间是互相独立的。上式其实就是一个简单的一元二次式，我们很容易求出各个叶子节点的最佳值以及此时目标函数的值。如下所示：
$$
\begin{align*}
w_j^* &= -\frac{G_j}{H_j+\lambda} \\
 obj^*&=-\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma T
\end{align*}
$$
$obj^*$表示了这棵树的结构有多好，值越小，代表这样结构越好！也就是说，它是衡量第t棵CART树的结构好坏的标准。这个值仅仅是用来衡量结构的好坏的，与叶子节点的值是无关的，因为obj*只和Gj和Hj和T有关，而它们又只和树的结构(q(x))有关，与叶子节点的值可是半毛关系没有。

Note：这里，我们对$w_j^*$给出一个直觉的解释，以便能获得感性的认识。我们假设分到j这个叶子节点上的样本只有一个。那么，$w_j^*$就变成如下这个样子：
$$
w_j^* =\underbrace{ \frac{1}{h_j+\lambda}}_{学习率}\cdot \underbrace{-g_j}_{负梯度}
$$
这个式子告诉我们，$w_j^*​$的最佳值就是负的梯度乘以一个权重系数，该系数类似于随机梯度下降中的学习率。观察这个权重系数，我们发现，$h_j​$越大，这个系数越小，也就是学习率越小。$h_j​$越大代表什么意思呢？代表在该点附近梯度变化非常剧烈，可能只要一点点的改变，梯度就从10000变到了1，所以，此时，我们在使用反向梯度更新时步子就要小而又小，也就是权重系数要更小。

 ### 找出最优树结构

好了，有了评判树的结构好坏的标准，我们就可以先求最佳的树结构，这个定出来后，最佳的叶子结点的值实际上在上面已经求出来了。

问题是：树的结构近乎无限多，一个一个去测算它们的好坏程度，然后再取最好的显然是不现实的。所以，我们仍然需要采取一点策略，这就是逐步学习出最佳的树结构。这与我们将M棵树的模型分解成一棵一棵树来学习是一个道理，只不过从一棵一棵树变成了一层一层节点而已。

具体来说，对于特征集中每一个特征$a$， 找出该特征的所有切分点，对每一个确定的切分点，我们衡量切分好坏的标准如下：
$$
Gain=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma
$$
这个Gain实际上就是单节点的obj减去切分后的两个节点的树obj，Gain如果是正的，并且值越大，表示切分后obj越小于单节点的obj，就越值得切分。同时，我们还可以观察到，Gain的左半部分如果小于右侧的$\gamma$，则Gain就是负的，表明切分后obj反而变大了。$\gamma$在这里实际上是一个临界值，它的值越大，表示我们对切分后obj下降幅度要求越严。这个值也是可以在xgboost中设定的。

扫描结束后，我们就可以确定是否切分，如果切分，对切分出来的两个节点，递归地调用这个切分过程，我们就能获得一个相对较好的树结构。

**注意：**xgboost的切分操作和普通的决策树切分过程是不一样的。普通的决策树在切分的时候并不考虑树的复杂度，而依赖后续的剪枝操作来控制。xgboost在切分的时候就已经考虑了树的复杂度，就是那个γ参数。所以，它不需要进行单独的剪枝操作。  

## 参考

[xgboost的原理没你想象的那么难](https://www.jianshu.com/p/7467e616f227)

[怎样理解xgboost能处理缺失值?](https://www.zhihu.com/question/58230411)

## 思考

Q:xgb怎么处理缺失值？

A：xgb处理缺失值的方法和其他树模型不同,xgboost把缺失值当做稀疏矩阵来对待，本身的在节点分裂时不考虑的缺失值的数值。缺失值数据会被分到左子树和右子树分别计算损失，选择较优的那一个。如果训练中没有数据缺失，预测时出现了数据缺失，那么默认被分类到右子树。

PS:随机森林怎么处理缺失值？

1. 数值型变量用中值代替，类别型变量用众数代替。
2. 引入了权重。即对需要替换的数据先和其他数据做相似度测量,补全缺失点是相似的点的数据会有更高的权重W

Q:xgb为什么用二阶展开项？

A:

Q:如何保证一元二次式（7）是开口向上的呢？即如果$\frac{1}{2}(H_j+\lambda)<0$,那么不是岂可以使损失函数达到无限小？