---
layout: post
title: "因子机深入解析"
description: ""
category: "machine-learning"
tags: ["机器学习","算法","因子机"]
---

## 关于
在组内做过一次因子机的技术分享，这里记录的是分享的一些内容。

## 综述
2010年，日本大阪大学(Osaka University)的 Steffen Rendle 在矩阵分解(MF)、SVD++[2]、PITF[3]、FPMC[4]
等基础之上，归纳出针对高维稀疏数据的因子机(Factorization Machine, FM)模型。因子机模型可以将上述模型全部纳入一个统一的框架进行分析。并且，Steffen Rendle 实现了一个单机多线程版本的 [libFM](http://www.libfm.org/)。在随后的 [KDD Cup 2012，track2 广告点击率预估(pCTR)](https://www.kaggle.com/c/kddcup2012-track2)中，国立台湾大学[4]和 Opera Solutions[5] 两只队伍都采用了 FM，并获得大赛的冠亚军而使得 FM 名声大噪。随后，台湾大学的 Yuchin Juan 等人在总结自己在两次比赛中的经验以及 Opera Solutions 队伍中使用的 FM 模型的总结，提出了一般化的 FFM 模型[6]，并实现了单机多线程版的 [libFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)，并做了深入的试验研究。事实上，Opera Solutions 在比赛中用的 FM 就是FFM。FM 模型只考虑了 user 和 iterm 间的交互，FFM 考虑更多id间的交互，是一种更一般化的模型。

## 什么是因子机
### 因子机的定义
机器学习中的建模问题可以归纳为从数据中学习一个函数 $$f: R^n \rightarrow T$$，它将实值的特征向量 $$x \in R^n$$
映射到一个特定的集合中。例如，对于回归问题，集合 T 就是实数集 R，对于二分类问题，这个集合可以是 $$\{+1, -1\}$$.
对于监督学习，通常有一标注的训练样本集合 $$D = \{(x^{(1)},y^{(1)}),..., (x^{(n)},y^{(n)})\}$$。

线性函数是最简单的建模函数，它假定这个函数可以用参数 $$w$$ 来刻画，

$$
\phi(x) = w_0 + \sum_i w_i x_i
$$

对于回归问题，$$y = \phi(x) $$；而对于二分类问题，需要做对数几率函数变换（逻辑回归）

$$
y = \frac{1}{1 + \exp{-\phi(x)}}
$$

线性模型的缺点是无法学到模型之间的交互，而这在推荐和CTR预估中是比较关键的。例如，CTR预估中常将用户id和广告id onehot
编码后作为特征向量的一部分。

为了学习特征间的交叉，SVM通过多项式核函数来实现特征的交叉，实际上和多项式模型是一样的，这里以二阶多项式模型为例

$$
\phi(x) = w_0 + \sum_i w_i x_i + \sum_i \sum_{j \le i} w_{ij} x_i x_j \\\\
        = w_0 + \mathbf{w^1}^T \mathbf{x} + \mathbf{x}^T \mathbf{W^2} \mathbf{x}
$$

多项式模型的问题在于二阶项的参数过多，设特征维数为 $$n$$，那么二阶项的参数数目为 $$ n(n+1)/2 $$。
对于广告点击率预估问题，由于存在大量id特征，导致 $$n$$ 可能为 $$10^7$$维，这样一来，模型参数的
量级为 $$10^{14}$$，这比样本量 $$4\times 10^7$$多得多[6]！这导致只有极少数的二阶组合模式才能在样本中找到，
而绝大多数模式在样本中找不到，因而模型无法学出对应的权重。例如，对于某个$$w_{ij}$$，样本中找不到$$x_i=1,x_j=1$$
（这里假定所有的特征都是离散的特征，只取0和1两个值）这种样本，那么$$w_{ij}$$恒为0，从而导致参数学习失败！

很容易想到，可以对二阶项参数施加某种限制，减少模型参数的自由度。FM 施加的限制是要求二阶项系数矩阵是低秩的，能够分解为低秩矩阵的乘积

$$
\mathbf{W^2} = \mathbf{V}^T \mathbf{V}, V \in \mathbb{R}^{k \times n} \\\\
w_{ij} = \mathbf{v_i}^T \mathbf{v_j} , \mathbf{v_i} \in \mathbb{R}^{k} \\\\
\mathbf{V} = [\mathbf{v_1}, ..., \mathbf{v_n}]
$$

这样一来，就将参数个数减少到 $$kn$$，可以设置较少的k值（一般设置在100以内，$$k << n$$），极大地减少模型参数，增强模型泛化能力，这跟矩阵分解的方法是一样的。向量 $$\mathbf{v_i}$$ 可以解释为第$$i$$个特征对应的隐因子或隐向量。
以user和item的推荐问题为例，如果该特征是user，可以解释为用户向量，如果是item，可以解释为物品向量。

### 计算复杂度
因为引入和二阶项，如果直接计算，时间复杂度将是 $$O(\bar{n}^2)$$，$$\bar{n}$$是特征非零特征数目，
可以通过简单的数学技巧将时间复杂度减少到线性时间复杂度。

基于一个基本的观察，齐二次交叉项之和可以表达为平方和之差

$$
\sum_i \sum_{j \le i} z_i z_j = \frac{1}{2} \left( (\sum_i z_i)^2 - \sum_i z_i^2 \right)
$$

上式左边计算复杂度为$$O(n^2)$$，而右边是$$O(n)$$。根据上式，可以将原表达式中二次项化简为

$$
\sum_i \sum_{j \le i} w_{ij} x_i x_j = \sum_i \sum_{j \le i} \sum_k v_{ik} v_{jk} x_i x_j \\\\
= \frac{1}{2} \sum_k  \left( (\sum_i v_{ik} x_i)^2 - \sum_i v_{ik}x_i^2 \right)
$$

上式计算时间复杂度是$$O(\bar{n})$$。

基于梯度的优化都需要计算目标函数对参数的梯度，对FM而言，目标函数对参数的梯度可以利用链式求导法则分解为目标函数对$$\phi$$的梯度和$$\frac{\partial \phi}{\partial \theta}$$ 的乘积。前者依赖于具体任务，后者可以简单的求得

$$
\frac{\partial \phi}{\partial \theta} =
\begin{cases}
1, \\&  \text{if $$\theta$$ is $$w_0$$} \\\\
x_i, \\&  \text{if $$\theta$$ is $$w_i$$} \\\\
x_i\sum_j v_{jk} x_j - v_{ik}x_i^2, \\&  \text{if $$\theta$$ is $$w_{ik}$$}
\end{cases}
$$

### 优化方案
论文[2]中给出了三种优化方案，它们分别是

1. 随机梯度下降，这种方案收敛慢而且非常敏感，可以利用现代的一些trick，例如采用 AdaGrad 算法，采用自适应学习率，效果相对比较好，论文[6]对FFM就采用这种方案。
2. 交替方向乘子(ALS)，这种方案只适用于回归问题，它每次优化一个参数，把其他参数固定，好处是每次都是一个最小二乘问题，有解析解。
3. 基于蒙特卡罗马尔科夫链的优化方案，论文中效果最好的方案，细节可以参考原文。




## 参考文献
1. Y. Koren, “Factorization meets the neighborhood: a multifaceted collabo- rative filtering model,” in KDD ’08: Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. New York, NY, USA: ACM, 2008, pp. 426–434.
2. S. Rendle and L. Schmidt-Thieme, “Pairwise interaction tensor factoriza- tion for personalized tag recommendation,” in WSDM ’10: Proceedings of the third ACM international conference on Web search and data mining. New York, NY, USA: ACM, 2010, pp. 81–90.
3. S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, “Factorizing per- sonalized markov chains for next-basket recommendation,” in WWW ’10: Proceedings of the 19th international conference on World wide web. New York, NY, USA: ACM, 2010, pp. 811–820.
4. A two-stage ensemble of diverse models for advertisement ranking in KDD Cup 2012[C]//KDDCup. 2012.
5. Jahrer M, Toscher A, Lee J Y, et al. Ensemble of collaborative filtering and feature engineered models for click through rate prediction[C]//KDDCup Workshop. 2012.
6. Juan Y, Zhuang Y, Chin W, et al. Field-aware Factorization Machines for CTR Prediction[C]. conference on recommender systems, 2016: 43-50.
