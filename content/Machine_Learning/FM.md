---
title: "因子机FM"
layout: page
date: 2018-08-29 00:00
---

[TOC]

## FM
考虑线性模型$f(x)=w_0+\sum_{i}w_ix_i$，由于表达式中只有一阶项，无法学到特征之间的交互，
因此需要引入多项式模型，一般为了简单，会只捕捉二阶的关系，即特征间两两相互影响的关系，如下：
$$
\phi(x) = w_0 + \sum_i w_i x_i + \sum_i \sum_{j \lt i} w_{ij} x_i x_j \\\\
        = w_0 + \mathbf{w_1}^T \mathbf{x} + \mathbf{x}^T \mathbf{W_2} \mathbf{x}
$$
这里每两个特征有一个参数w要学习。
这里仍有问题，对于二项式回归来说，如果有n个特征，那么要学习到两两之间的关系，有n(n−1)/2个参数要去学习，对于实际中的复杂任务来说，n的值往往特别大，会造成要学习的参数特别多的问题。 
同时，又由于实际数据会有稀疏性问题，有些特征两两同时不为0的情况很少，当一个数据中任何一个特征值为0的时候，那么其他特征与此特征的相互关系将没有办法学习。
受到矩阵分解的启发，为了解决上述两个问题，引入了因子分解机。
基于矩阵分解的思想，将二阶项系数矩阵分解为两个低阶矩阵的乘积，
$$
\mathbf{W} = \mathbf{V}^T \mathbf{V}, V \in \mathbb{R}^{k \times n} \\\\
w_{ij} = \mathbf{v_i}^T \mathbf{v_j} , \mathbf{v_i} \in \mathbb{R}^{k} \\\\
\mathbf{V} = [\mathbf{v_1}, ..., \mathbf{v_n}]
$$
通过分解就将参数个数减少到$ kn$，向量$\mathbf{v_i}$可以解释为第i个特征对应的隐向量。 
### 计算复杂度
直接计算$ \sum_i \sum_{j \lt i}  \sum_k v_{ik}v_{jk} x_i x_j $的复杂度是$O(kn^2)$，n为非零特征的个数。而通过一个简单的变换
$$
\sum_i \sum_{j \lt i}  \sum_k v_{ik}v_{jk} x_i x_j = \frac{1}{2} \sum_k  \left( \left(\sum_i v_{ik} x_i \right)^2 - \sum_i v_{ik}^2 x_i^2 \right)
$$
就可以将复杂度降低到$O(kn)$
基于梯度的优化都需要计算目标函数对参数的梯度，对FM而言，目标函数对参数的梯度可以利用链式求导法则分解为目标函数对$ϕ$的梯度和$\frac{\partial \phi}{\partial \theta}$的乘积。前者依赖于具体任务，后者可以简单的求得
$$
\frac{\partial \phi}{\partial \theta} =
\begin{cases}
1, &  \text{if $\theta$ is $w_0$} \\\\
x_i, &  \text{if $\theta$ is $w_i$} \\\\
x_i\sum_j v_{jk} x_j - v_{ik}x_i^2, &  \text{if $\theta$ is $v_{ik}$}
\end{cases} %]]>
$$
### 学习方法
1. 随机梯度下降
2. 交替最小二乘法（Alternating least-squares）
3. 马尔科夫链蒙特卡洛法MCMC

## FFM
在实际预测任务中，特征往往包含多种id，如果不同id组合时采用不同的隐向量，那么这就是 FFM(Field Factorization Machine) 模型。它将特征按照事先的规则分为多个场(Field)，特征
$x_i$属于某个特定的场$f$。每个特征将被映射为多个隐向量$\mathbf{v}\_{i1},...,\mathbf{v}\_{if}$，每个隐向量对应一个场。当两个特征$xi,xj$组合时，用对方对应的场对应的隐向量做内积。
$$
w_{ij} = \mathbf{v}\_{i,f_j}^T\mathbf{v}\_{j,f_i}
$$
$f_i,f_j$分别是特征$xi,xj$对应的场编号。FFM 由于引入了场，使得每两组特征交叉的隐向量都是独立的，可以取得更好的组合效果，但是使得计算复杂度无法通过优化变成线性时间复杂度，每个样本预测的时间复杂度为 $O(n^2k)$，不过FFM的k值通常远小于FM的k值。FM 可以看做只有一个场的 FFM。

## 参考
[因子机深入解析](https://tracholar.github.io/machine-learning/2017/03/10/factorization-machine.html)
[FM](https://blog.csdn.net/liruihongbob/article/details/75008666)
