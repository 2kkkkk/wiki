---
title: "伯努利分布 二项分布 泊松分布"
layout: page
date: 2018-07-06 00:00
---

[TOC]

## 伯努利分布

首先说伯努利分布， 这个是最简单的分布，就是0-1分布

以抛硬币为例， 为正面的概率为p， 反面的概率为q，是一种离散型概率分布，也是很多分布的基础

## 二项分布

以伯努利分布为基础，假设伯努利分布中得1的概率为p, 0的概率为q，那么二项分布求的就是进行n次伯努利分布，得到k次1的概率是多少

即n次独立重复试验

## 泊松分布

知乎上有一个回答用了一个例子生动地解释了泊松分布。

楼下有一家馒头店，馒头店老板统计了一周每日卖出的馒头

|      | 卖出个数 |
| ---- | -------- |
| 周一 | 3        |
| 周二 | 7        |
| 周三 | 4        |
| 周四 | 6        |
| 周五 | 5        |

每天早上六点到十点营业，生意挺好，就是发愁一个事情，应该准备多少个馒头才能既不浪费又能充分供应？

按道理讲均值是不错的选择，但是如果每天准备5个馒头的话，从统计表来看，至少有两天不够卖。

老板尝试把营业时间抽象为一根线段，把这段时间用$T$来表示：

<img src="/wiki/static/images/adaboost/p1.png" alt="joey"/>

然后把周一的三个馒头按照销售时间放在线段上：

<img src="/wiki/static/images/adaboost/p2.png" alt="joey"/>

把$T$均分为四个时间段：

<img src="/wiki/static/images/adaboost/p3.png" alt="joey"/>

此时，在每一个时间段上，要不卖出了（一个）馒头，要不没有卖出：

<img src="/wiki/static/images/adaboost/p4.png" alt="joey"/>

在每个时间段，就有点像抛硬币，要不是正面（卖出），要不是反面（没有卖出）。

$T$ 内卖出3个馒头的概率，就和抛了4次硬币（4个时间段），其中3次正面（卖出3个）的概率一样了。

这样的概率通过二项分布来计算就是：
$$
\binom{4}{3}p^3(1-p)^1
$$
但是，如果把周二的七个馒头放在线段上，分成四段就不够了：每个时间段，有卖出3个的，有卖出2个的，有卖出1个的，就不再是单纯的“卖出、没卖出”了。不能套用二项分布了。

解决这个问题也很简单，把$T$ 分为20个时间段，那么每个时间段就又变为了抛硬币，这样，$T$时间内卖出7个馒头的概率就是（相当于抛了20次硬币，出现7次正面）：
$$
\binom{20}{7}p^{7}(1-p)^{13}
$$
 为了保证在一个时间段内只会发生“卖出、没卖出”，干脆把时间切成n份，越细越好，用极限来表示：
$$
\lim_{n\rightarrow \infty}\binom{n}{7}p^{7}(1-p)^{n-7}
$$
更抽象一点，$T$时刻内卖出$k$个馒头的概率为：
$$
\lim_{n\rightarrow \infty}\binom{n}{k}p^{k}(1-p)^{n-k}
$$
老板用笔敲了敲桌子，“只剩下一个问题，概率$p$怎么求？”

在上面的假设下，问题已经被转为了二项分布。二项分布的期望为：
$$
E(X)=np=\mu
$$
有了$p=\frac{\mu}{n}$后，可以算出
$$
\lim_{n\rightarrow \infty}\binom{n}{k}(\frac{\mu}{n})^{k}(1-\frac{\mu}{n})^{n-k}=\frac{\mu^k}{k!}e^{-\mu}
$$
也就是说，在$T$时间内卖出$k$个馒头的概率为：
$$
P(X=k)=\frac{\mu^k}{k!}e^{-\mu}
$$
一般来说，我们会换一个符号，让$\mu=\lambda$，所以：
$$
P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda}
$$
这就是教科书中的泊松分布的概率密度函数。

老板依然蹙眉，不知道$\mu$ 啊？

没关系，可以用样本均值来近似：
$$
\bar{X}=5\approx \mu
$$
于是，
$$
P(X=k)=\frac{5^k}{k!}e^{-5}
$$
画出概率密度函数的曲线，可以看到，如果每天准备8个馒头的话，那么足够卖的概率就是把前8个的概率加起来：

<img src="/wiki/static/images/adaboost/p5.png" alt="joey"/>

这样93%的情况够用，偶尔卖缺货也有助于品牌形象。

老板算出一脑门的汗，“那就这么定了！”

这个故事告诉我们，要努力学习啊，要不以后馒头都没得卖。

## 参考

https://www.zhihu.com/question/26441147 

