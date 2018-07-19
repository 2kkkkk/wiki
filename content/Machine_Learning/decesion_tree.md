---
title: "Decision Tree"
layout: page
date: 2018-07-08 00:00
---

[TOC]

## Decision Tree Hypothesis and Algorithm

​	之前介绍了很多aggregation model，aggregation的核心就是将许多可供选择使用的比较好的hypothesis融合起来，利用集体的智慧组合成G，使其得到更好的机器学习预测模型。下面，我们先来看看已经介绍过的aggregation type有哪些。

| Aggregation type | blending         | Learning      |
| ---------------- | ---------------- | ------------- |
| uniform          | voting/averaging | bagging       |
| Non-uniform      | linear           | adaboost      |
| conditional      | stacking         | decision tree |

​	aggregation type有三种：uniform，non-uniform，conditional。它有两种情况，一种是所有的g是已知的，即blending。对应的三种类型分别是voting/averaging，linear和stacking。另外一种情况是所有g未知，只能通过手上的资料重构g，即learning。其中uniform和non-uniform分别对应的是Bagging和AdaBoost算法，而conditional对应的就是我们将要介绍的Decision Tree算法。

​	决策树（Decision Tree）模型是一种传统的算法，它的处理方式与人类思维十分相似。例如下面这个例子，对下班时间、约会情况、提交截止时间这些条件进行判断，从而决定是否要进行在线课程测试。如下图所示，整个流程类似一个树状结构。

<img src="/wiki/static/images/adaboost/human.png" alt="human"/>

​	图中每个条件和选择都决定了最终的结果，Y or N。蓝色的圆圈表示树的叶子，即最终的决定。那么，如何来描述这棵树呢，把这种树状结构对应到一个hypothesis $G(x)$中，$G(x)$的表达式为：
$$
G(x)=\sum_{t=1}^{T}q_t(x)\cdot g_t(x)
$$
​	 $G(x)$由许多 $g_t(x)$ 组成，即aggregation的做法。每个$g_t(x)$就代表上图中的蓝色圆圈（树的叶子）。这里的$g_t(x)$是常数，因为是处理简单的classification问题。我们把这些$g_t(x)$称为base hypothesis。$q_t(x)$表示每个$g_t(x)$成立的条件，代表上图中橘色箭头的部分，即从根节点出发到相应叶子节点的路径。不同的$g_t(x)$对应于不同的$q_t(x)$，即从树的根部到底端叶子的路径不同。图中中的菱形代表每个简单的节点。所以，这些base hypothesis和conditions就构成了整个$G(x)$的形式，就像一棵树一样，从根部到顶端所有的叶子都安全映射到上述公式上去了。

​	如果从另外一个方面来看决策树的形式，不同于上述$G(x)$的公式，我们可以利用条件分支的思想，将整体$G(x)$分成若干个$G_{c}(x)$，也就是把整个大树分成若干个小树，如下所示：
$$
G(x)=\sum_{c=1}^{C}\left [ b(x)=c \right ]\cdot G_{c}(x)
$$
​	上式中，$G(x)$表示完整的大树，即full-tree hypothesis，$b(x)$表示每个分支条件，即branching criteria，$G_{c}(x)$表示第c个分支下的子树，即sub-tree。这种结构被称为递归型的数据结构，即将大树分割成不同的小树，再将小树继续分割成更小的子树。所以，决策树可以分为两部分：root和sub-trees。

​	在详细推导决策树算法之前，我们先来看一看它的优点和缺点。

| 优点                         | 缺点                                     |
| ---------------------------- | ---------------------------------------- |
| 模型直观，便于理解，应用广泛 | 缺少足够的理论支持                       |
| 算法简单，容易实现           | 如何选择合适的树结构对初学者来说比较困惑 |
| 训练和预测时，效率较高       | 决策树代表性的演算法比较少               |

​	决策树的递归表达形式为
$$
G(x)=\sum_{c=1}^{C}\left [ b(x)=c \right ]\cdot G_{c}(x)
$$
​	那么一个基本的决策树算法的流程如下，

------

1. for .....
2. ​	从特征集A中选择最优划分特征$a_{*}$
3. ​	for **$a_{*}$的每一个取值$a_{*}^{v}$**:
4. ​		为结点生成一个分支，令$D_v$表示$D$中在$a_{*}$中取值为$a_{*}^{v}$的样本子集
5. ​		if $D_v$为空：将分支结点标记为叶结点，其类别标记为$D$中样本最多的
6. ​                else： 以($D_v$，$A-a_{*}$)为分支结点继续划分

------

决策树学习算法包括决策树的生成、特征选择、决策树的修剪。从上面的算法流程中可以看出，决策树学习的关键是第2行，即如何从特征集A中选择最优划分特征$a_{*}$，一般而言，随着划分过程的不断进行，我们希望决策树的分支结点所包含的样本尽可能属于同一类别，即结点的“纯度”越来越高。

## 特征选择

### 信息增益 

​	“信息熵”是度量样本集合纯度最常用的一种指标。假定样本集合$D$中第k类样本所占的比例为$p_{k}(k=1,2...,\left | \mathcal{Y} \right |)$，则$D$的信息熵定义为
$$
Ent(D)=-\sum_{k=1}^{|\mathcal{Y} |}p_{k}\cdot {log_{2}}^{p_{k}}
$$
$Ent(D)$的值越小，$D$的纯度越高。

于是可以计算出用属性$a$对样本集$D$进行划分所获得的“信息增益”
$$
Gain(D,a)=Ent(D)-\sum_{v=1}^{|V|}\frac{D^{v}}{D}Ent(D^{v})
$$
信息增益即训练数据集中类与特征的互信息。一般而言，信息增益越大，则意味着使用属性$a$来进行划分所获得的“纯度提升”越大，即在上面算法第2步选择属性
$$
a_{*}=\mathop{\arg\max}_{a\in A}Gain(D,a)
$$
著名的ID3算法就是以信息增益为准则来选择划分属性。

### 信息增益率 

但是，如果将样本编号也作为一个属性的话，可以算出它的信息增益远大于其他候选划分属性，例如，样本集有17个样本，那么“编号”这个属性将产生17个分支，每个分支结点仅包含一个样本，这些分支结点的纯度已经达到最大，然而，这样的决策树显然不具有泛化能力。

实际上，信息增益准则对可取值数目较多的属性有所偏好，为减少这种偏好可能带来的不利影响，可以用“增益率”来选择最优划分属性，定义为
$$
Gain_ratio=\frac{Gain(D,a)}{IV(a)}
$$
其中
$$
IV(a)=-\sum_{v=1}^{|V|}\frac{|D^{v}|}{D}{log_{2}}^{}\frac{|D^{v}|}{D}
$$
称为属性$a$的“固有值”，属性$a$的可能取值数目越多（即V越大），$IV(a)$的值通常会越大，需要注意的是，增益率准则对可取值数目较少的属性有所偏好，因此，著名的C4.5算法并不是直接选择增益率最大的候选划分属性，而是使用了一个启发式：先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的。

### 基尼指数

著名的CART决策树使用“基尼指数”来选择划分属性，数据集$D$的纯度可用基尼指数来度量：
$$
Gini(D)=\sum_{k=1}^{|\mathcal{Y} |}\sum_{k^{'}\neq k}p_kp_{k^{'}}=1=\sum_{k=1}^{|\mathcal{Y} |}p_k^{2}
$$
直观上来说，$Gini(D)$反映了从数据集$D$中随机抽取两个样本，其类别标记不一致的概率，因此$Gini(D)$越小，数据集$D$的纯度越高。

类似于式（5），属性$a$的基尼指数定义为
$$
Gini_index(D,a)=\sum_{v=1}^{|V|}\frac{|D^{v}|}{D}Gini(D^{v})
$$
即最优划分属性
$$
a_{*}=\mathop{\arg\max}_{a\in A}Gain(D,a)
$$

## 决策树剪枝

​	决策树算法递归的产生决策树，但很容易过拟合，原因在于学习时过多的考虑如何提高对训练数据的正确分类，从而构建出过于复杂的决策树，解决这个问题的方法是剪枝。剪枝的基本策略有预剪枝和后剪枝。预剪枝是指在决策树的生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能的提升，则停止划分并将当前结点标记为叶子结点；后剪枝则是先从训练集生成一棵完整的决策树，然后自底向上的对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来泛华性能的提升，则将该子树替换为叶结点。

​	如何判断决策树泛化性能是否提升呢？可以用留出法，预留一部分数据用作“验证集”来进行评估。

​	预剪枝使得决策树很多分支都没有展开，这不仅降低了过拟合的风险，还显著减少了决策树的训练和预测时间开销，但另一方面，有些分支的当前划分虽然不能提升泛化性能，但在其基础上进行的后续划分却有可能导致性能显著提高，预剪枝基于“贪心”本质禁止这些分支展开，给预剪枝决策树带来了**欠拟合**的风险。

​	后剪枝的欠拟合风险很小，泛华性能往往优于预剪枝，但后剪枝是在生成完全决策树之后进行的，并且要自底向上地对树中所有非叶子结点进行逐一考察，因此其**训练时间开销**要大得多。

​	下面从理论上分析后剪枝算法。决策树的剪枝往往通过极小化决策树整体的损失函数来实现。设树$T$的叶结点个数为$|T|$，$t$是树$T$ 的叶结点，该叶结点有$N_{t}$ 个样本点，其中第k类的样本点有$N_{tk}$个，$k=1,2...,K$，$H_{t}(T)$为叶结点t上的经验熵，$\alpha \geq 0$为参数，则决策树的损失函数可以定义为
$$
C_{\alpha }(T)=\sum_{t=1}^{T}N_{t}H_{t}(T)+\alpha |T|
$$
其中经验熵为
$$
H_{t}(T)=-\sum_{k}\frac{N_{tk}}{N_{t}}log\frac{N_{tk}}{N_{t}}
$$
将式（12）右端的第一项记做$C(T)$

这时有
$$
C_{\alpha }(T)=C(T)+\alpha |T|
$$
上式中，$C(T)$表示模型对训练数据的预测误差，即模型与训练数据的拟合程度，$|T|$表示模型复杂度，参数$\alpha \geq 0$控制两者之间的影响。较大的$\alpha$促使选择较简单的树，较小的$\alpha$促使选择较复杂的树。$\alpha$取值多少可以通过validation来确定。利用式（12）定义的损失函数最小原则进行剪枝就是用正则化的极大似然估计进行模型选择。具体的剪枝算法为

------

输入：生成算法产生的整个树$T$ ，参数$alpha$

输出：修剪后的子树$T_{\alpha}$

1. 计算每个结点的经验熵
2. 递归地从树的叶结点向上回缩，若**一组**叶结点回缩到父节点之前与之后的整体树分别为$T_{B}$与$T_{A}$，且$C_{\alpha }(T_{A})\leq C_{\alpha }(T_{B})$，则进行剪枝，即将父节点变为新的叶结点
3. 返回第2步，直至不能继续为止，得到损失函数最小的子树$T_{\alpha}$

------

## 连续值与缺失值

### 连续值

由于连续特征的可取值数目不再有限，因此，不能直接根据连续特征的可取值来对结点进行划分，最简单的策略是是采用二分法进行处理。

给定样本集$D$和连续特征$a$，假定$a$在$D$上出现了n个不同的取值，将这些值从小到大进行排序，记为${a^{1},a^{2},a^{3}...,a^{n}}$。基于划分点t可将$D$分为子集$D_{t}^{+}$和$D_{t}^{-}$，其中$D_{t}^{-}$包含那些在属性$a$上取值不大于t的样本，而$D_{t}^{+}$包含那些在属性$a$上取值大于t的样本。显然，对相邻的属性取值$a^{i}$和$a^{i+1}$来说，t在区间$[a^{i},a^{i+1})$中取任意值所产生的划分结果一致，因此，对连续属性$a$，我们考察包含n-1个元素的候选划分点集合
$$
T_{a}=\left \{ \frac{a^{i}+a^{i+1}}{2} |1\leqslant  i\leqslant n-1|\right \}
$$
即把区间$[a^{i},a^{i+1})$的中位点$\frac{a^{i}+a^{i+1}}{2} $作为候选划分点。然后，我们就可以像离散属性值一样来考虑这些划分点。

**需要注意的是，与离散属性不同，若当前结点划分属性为连续属性，该属性还可以作为其后代结点的划分属性**

### 缺失值

 	我们需要解决两个问题，（1）如何在属性值缺失的情况下进行最优划分属性选择（2）给定划分属性，若样本在该属性的值缺失，如何对样本进行划分

​	给定训练集D和属性$a$，令$\tilde{D}$表示D中在属性$a$上没有缺失值得样本子集，对于问题（1），显然仅可根据$\tilde{D}$来判断属性$a$的优劣。假定属性$a$有V个取值${a^{1},a^{2},...,a^{V}}$，令$\tilde{D}^{v}$表示$\tilde{D}$中在属性$a$上取值为$a^{v}$的样本子集，$\tilde{D}_{k}$表示$\tilde{D}$中属于第k类$(k=1,2...,\left | \mathcal{Y} \right |)$的样本子集，则显然有$\tilde{D}=\bigcup _{k=1}^{ \mathcal{|Y|}}\tilde{D}_{k}$，$\tilde{D}=\bigcup _{v=1}^{ \mathcal{|V|}}\tilde{D}^{v}$，假定我们为每个样本$\mathbf{x}$赋予一个权重$w_\mathbf{x}$，并定义
$$
\rho =\frac{\sum_{x \in \tilde{D}}w_{\mathbf{x}}}{\sum_{x \in D}w_{\mathbf{x}}}
$$

$$
\tilde{p_{k}}=\frac{\sum_{x \in \tilde{D_{k}}}w_{\mathbf{x}}}{\sum_{x \in \tilde{D}}w_{\mathbf{x}}}\  \   \ \ \  (k=1,2...,\left | \mathcal{Y} \right |)
$$

$$
\tilde{r_{v}}==\frac{\sum_{x \in \tilde{D^{v}}}w_{\mathbf{x}}}{\sum_{x \in \tilde{D}}w_{\mathbf{x}}}\ \ (v=1,2...,\left | V \right |)
$$

直观的看，对属性$a$，$\rho$表示无缺失值样本所占的比例，$\tilde{p_{k}}$表示无缺失值样本中第k类所占的比例，$\tilde{r_{v}}$表示无缺失值样本中在属性$a$上取值$a^{v}$的比例，显然，$\sum _{k=1}^{\mathcal{Y}}\tilde{p_{k}}=1,\sum _{v=1}^{V}\tilde{r_{v}}=1$.

​	因此，我们可以将信息增益的计算式推广为
$$
\begin{align*}
 Gain(D,a) &= \rho \times Gain(\tilde{D},a)\\
 &=\rho \times (Ent(\tilde{D})-\sum_{v=1}^{V}\tilde{r_{v}}Ent(\tilde{D}^{v}))
\end{align*}
$$
其中，
$$
Ent(\tilde{D})=)-\sum_{k=1}^{{\mathcal{Y}}}\tilde{p_{k}}log_{2}^{\tilde{p_{k}}}
$$
​	对于问题（2)，若样本$\mathbf{x}$在划分属性$a$上的取值已知，则将$\mathbf{x}$划入与其取值对应的子节点，且样本权值在子结点中保持为$w_{\mathbf{x}}$。若样本在划分属性$a$上的取值未知，则将$\mathbf{x}$同时划入所有子结点，且样本权值在与属性值$a^{v}$对应的子结点中调整为$\tilde{r_{v}} \cdot w_{\mathbf{x}}$，直观的看，这就是让同一个 样本以不同的概率划入到不同的子结点中去。

## CART	

分类与回归树（CART）假定决策树是**二叉树**，内部结点的取值为“是”和"否"，左分支为取值为“是”的分支，右分支为取值为“否”的分支。这样的决策树等价于递归地二分每个特征，将输入空间即特征空间划分为有限个单元，并在这些单元上确定预测的概率分布，也就是在输入给定条件下输出的条件概率分布。

### CART回归树的生成

决策树的生成就是递归地构建二叉决策树的过程，对回归树用平方误差最小化准则，对分类树用基尼指数最小化准则，进行特征选择，生成二叉树。

最小二乘回归树生成算法

------

输入：训练集D

输出：回归树$f(x)$

(1) 在训练数据集所在的输入空间中，递归地将每个区域划分为两个子区域并决定每个子区域上的输出值

​	选择最优切分变量$j$和切分点$s$，求解

1. $$
   \mathop{\min}_{j,s}\left [ \mathop{\min}_{c_{1}}\sum_{x_{i}\in R_{1}(j,s)}(y_{i}-c_{1})^{2}+\mathop{\min}_{c_{2}}\sum_{x_{i}\in R_{2}(j,s)}(y_{i}-c_{2})^{2}\right ]
   $$

   遍历变量$j$，对固定的切分变量$j$扫描切分点$s$，选择使式(20)达到最大值的对$(j,s$)

(2) 用选定的对$(j,s$)划分区域并决定相应的输出值：
$$
R_{1}(j,s)=\left \{  x|x^{(j)}\leqslant s\right \},R_{2}(j,s)=\left \{  x|x^{(j)}> s\right \}
$$

$$
\hat{{c}}_{1}=\frac{1}{N_{1}}\sum_{x_{i}\in R_{1}(j,s)}y_{i},\hat{{c}}_{2}=\frac{1}{N_{2}}\sum_{x_{i}\in R_{2}(j,s)}y_{i}
$$

(3) 继续对两个子区域调用步骤（1）（2），直至满足停止条件

(4) 将输入空间划分为M个区域$R_{1},R_{2},..,R_{M}$，生成决策树：
$$
f(x)=\sum_{m=1}^{M}\hat{{c}}_{m}I(x\in R_{m})
$$

------

### CART 分类树的生成

分类树用基尼指数选择最优特征，同事决定该特征的最优二值切分点

CART分类树生成算法

------

根据训练集，从根结点开始，递归地对每个结点进行以下操作，构建二叉树

（1）设结点的训练数据集为$D$，对每一个特征$a$，对其可能取的每个值$a^{v}$，根据样本点对$a=a^{v}$的测试为“是”或“否”将$D$分割为$D_{1}$和$D_{2}$两部分，计算$a=a^{v}$时的基尼指数

（2）在所有可能的特征$a$中，选择基尼指数小的特征及其对应的切分点作为最优特征与最优切分点。依最优特征和最优切分点，从现结点生成两个子结点，将训练数据集依特征分配到两个子结点中去

（3）对两个子结点递归地调用（1）（2），直至满足停止条件

（4）生成CART决策树

算法停止计算的条件是结点中的样本个数小于预定阈值，或样本集的基尼指数小于预定阈值（样本基本属于同一类），或者没有更多特征

------

### CART剪枝

1. 剪枝，形成一个子树序列

在剪枝过程中，计算子树的损失函数：
$$
C_{\alpha}(T)=C(T)+\alpha|T|
$$
其中，$T$为任意子树，$C(T)$为对训练集的预测误差（如基尼指数），$|T|$为子树的叶结点个数，$C_{\alpha}(T)$为参数为$\alpha$时的子树$T$的整体损失，参数$\alpha$权衡训练数据的拟合程度与模型的复杂度。

**注意，我的理解是，这里的子树$T$指的是将整棵树$T_{0}$的某一内部结点$t$替换为叶子结点后形成的树，而不是以$t$为根结点的子树。这样想更容易理解一些**

Breiman等人证明，可以用递归的方法对数进行剪枝。将$\alpha$从小增大，$0=\alpha_{0}< \alpha_{1}<...<\alpha_{n}<+\infty $，产生一系列的区间$[\alpha_{i},\alpha_{i+1}),i=0,1,...,n$的最优子树序列{$T_{0},T_{1},...,T_{n}$}，序列中的子树是嵌套的。

具体的，从整体树$T_{0}$开始，对$T_{0}$的任意内部结点$t$，将$t$的所有子结点回缩，使结点$t$变为叶子结点，这样操作后所生成的子树记为$T_{t}$，$T_{t}$的损失函数为
$$
C_{\alpha}(T_{t})=C(T_{t})+\alpha|T_{t}|
$$
未剪枝前整体树$T_{0}$的损失函数为
$$
C_{\alpha}(T_{0})=C(T_{0})+\alpha|T_{0}|
$$
当$\alpha=0$以及$\alpha$非常小的时候，有不等式
$$
C_{\alpha}(T_{0})<C_{\alpha}(T_{t})
$$
当$\alpha$增大时，在某一$\alpha$有，
$$
C_{\alpha}(T_{0})=C_{\alpha}(T_{t})
$$
当$\alpha$再增大时，不等式（27）反向，只要$\alpha=\frac{C(T_{t})-C(T_{0})}{|T_{0}-T_{t}|}$，$T_{0}$和$T_{t}$有相同的损失函数值，但是$T_{t}$的结点少，因此$T_{t}$比$T_{0}$更可取，对$T_{0}$进行剪枝。

为此，对$T_{0}$中每一个内部结点t，计算
$$
g(t)=\frac{C(T_{t})-C(T_{0})}{|T_{0}-T_{t}|}
$$
它表示剪枝后整体损失函数减少的程度。将$g(t)$最小的$T_{t}$作为$T_{1}$，同时将最小的$g(t)$设为$\alpha_{1}$，$T_{1}$为区间$[\alpha_{1},\alpha_{2})$的最优子树。

如此剪枝下去，直至得到根结点。在这一过程中，不断增加$\alpha$的值，产生新的区间

2. 在剪枝得到的子树序列{$T_{0},T_{1},...,T_{n}$}中通过交叉验证选取最优子树$T_{\alpha}$

具体的，利用独立的验证集，测试子树序列{$T_{0},T_{1},...,T_{n}$}中各棵子树的平方误差或基尼指数，平方误差或基尼指数最小的决策树被认为是最优决策树。在子树序列{$T_{0},T_{1},...,T_{n}$}中，每棵子树都对应一个参数$\alpha_{1},\alpha_{2},...,\alpha_{n} $，所以当最优子树$T_{k}$确定时，对应的$\alpha_{k}$也确定了，即得到了最优决策树$T_{\alpha}$

## 思考

决策树的结构可能是二叉树，也可能是多叉树，取决于具体的算法

<img src="/wiki/static/images/adaboost/shu.png" alt="shu"/>

如上图，特征纹理可取值数目有3个，因此有3个分支，若规定决策树为二叉树的话（例如CART），那么结点“纹理=？”应该变成“纹理=清晰？”或者“纹理=稍糊？”或者“纹理=模糊？”，也就是将纹理这一个特征，分为“纹理=清晰？”“纹理=稍糊？”“纹理=模糊？”3个特征，然后进行最优特征选择，假定最优特征为“纹理=模糊？”，那么就根据样本点对“纹理=模糊？”的测试为“是”或“否”，将$D$分割为$D_{1}$和$D_{2}$两部分

## 参考

周志华 机器学习

李航 统计学习方法

红色石头机器学习之路

林轩田 机器学习技法