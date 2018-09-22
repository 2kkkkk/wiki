---
date: 2018-08-26 13:17
status: public
title: 'xgb split'
---

计算split的比较
对于连续值得feature，我们在求split时，一般是先做排序，然后顺序选择分割点并计算结构分数。
这里Boosted-Tree中的tree，相比于CART有一个很大的好处，那就是每一个sample的结构分是固定不变的，因此计算分数时只需要在上一次迭代后的分数的基础上算增量即可；而CART的impurity则无法计算增量，无论是熵还是gini系数，当增减sample时，需要全部重新算一次，这无疑是很大一部分的计算量。