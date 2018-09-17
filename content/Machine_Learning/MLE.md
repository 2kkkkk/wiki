---
title: "关于这个wiki"
layout: page
date: 2018-08-16 00:00
---

[TOC]

## 最大似然估计
找到参数$\theta$，使似然函数（观测数据$D$成立的可能性）最大，即
$$
\mathop{\arg\min}_{\theta} \ \ P(D|\theta)
$$
$P(x|\theta)$为似然函数。
## 最大后验估计
找到参数$\theta$，使后验概率最大，后验概率可以用贝叶斯公式表示：
$$
P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}
$$
$$
\mathop{\arg\min}_{\theta} \ \ \frac{P(D|\theta)P(\theta)}{P(D)}
$$
其中，分母$P(D)$与具体某一个$\theta$的选择无关，因为$P(D)=\int P(D,h)dh$，可以看做归一化常数，问题变成

$$
\mathop{\arg\min}_{\theta} \ \ P(x|\theta)P(\theta)
$$
可以看出，MAP与MLE的区别在于MAP考虑了参数的先验概率$P(\theta)$
## 贝叶斯估计
不考虑某一个最优的$\theta$，而是让所有的$\theta$都做预测，但最后通过一些加权平均的方式获得最终的结果。
$$
P(x|D)=\int P(x|\theta)P(\theta|D)d\theta
$$
其中，$x$为新的数据，$P(\theta|D)$可以看成是不同$\theta$的权重，$P(x|\theta)$是不同$\theta$的预测结果。贝叶斯估计做的事情就是得到不同$\theta$的权重。

## 参考
[机器学习中的MLE、MAP、贝叶斯估计](https://zhuanlan.zhihu.com/p/37215276)
[什么是最大似然估计、最大后验估计以及贝叶斯参数估计](http://www.anyv.net/index.php/article-2110343)