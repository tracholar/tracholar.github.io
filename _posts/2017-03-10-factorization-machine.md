---
layout: post
title: "因子机深入解析"
description: ""
category: "machine learning"
tags: ["机器学习","算法","因子机"]
---
[TOC]

## 关于
在组内做过一次因子机的技术分享，这里记录的是分享的一些内容。

## 综述
2010年，日本大阪大学(Osaka University)的 Steffen Rendle 在矩阵分解(MF)、SVD++[2]、PITF[3]、FPMC[4]
等基础之上，归纳出针对高维稀疏数据的因子机模型。因子机模型可以将上述模型全部纳入一个统一的框架进行分析。并且，
Steffen Rendle 实现了一个单机多线程版本的 [libFM](http://www.libfm.org/)。在随后的 [KDD Cup 2012，track2 广告点击率预估(pCTR)](https://www.kaggle.com/c/kddcup2012-track2)中，国立台湾大学[4]和 Opera Solutions[5] 两只队伍都采用了 FM，并获得大赛的冠亚军而使得 FM 名声大噪。随后，台湾大学的 Yuchin Juan 等人在总结自己在两次比赛中的经验以及 Opera Solutions 队伍中使用的 FM 模型的总结，提出了一般化的 FFM 模型，并实现了单机多线程版的 [libFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)，并做了深入的试验研究。事实上，Opera Solutions 在比赛中用的 FM 就是FFM。FM 模型只考虑了 user 和 iterm 间的交互，FFM 考虑更多id间的交互，是一种更一般化的模型。

## 什么是因子机
机器学习中的建模问题可以归纳为从数据中学习一个函数 $(f: R^n \righarrow T)$，它将实值的特征向量 $(x \in R^n)$
映射到一个特定的集合中。例如，对于回归问题，集合 T 就是实数集 R，对于二分类问题，这个集合可以是 $(\{+1, -1\})$.
对于监督学习，通常有一标注的训练样本集合 $(D = \{(x^{(1)},y^{(1)}),..., (x^{(n)},y^{(n)})\})$。

线性函数是最简单的建模函数，它假定这个函数可以用参数 $(w)$ 来刻画，

$$
\phi(x) = w_0 + \sum_i w_i x_i
$$

对于回归问题，$(y = \phi(x))$；而对于二分类问题，需要做对数几率函数变换（逻辑回归）

$$
y = \frac{1}{1 + \exp{-\phi(x)}}
$$

线性模型的缺点是无法学到模型之间的交互，而这在推荐和CTR预估中是比较关键的。例如，CTR预估中常将用户id和广告id onehot
编码后作为特征向量的一部分。

为了学习特征间的交叉



1. Y. Koren, “Factorization meets the neighborhood: a multifaceted collabo- rative filtering model,” in KDD ’08: Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. New York, NY, USA: ACM, 2008, pp. 426–434.
2. S. Rendle and L. Schmidt-Thieme, “Pairwise interaction tensor factoriza- tion for personalized tag recommendation,” in WSDM ’10: Proceedings of the third ACM international conference on Web search and data mining. New York, NY, USA: ACM, 2010, pp. 81–90.
3. S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, “Factorizing per- sonalized markov chains for next-basket recommendation,” in WWW ’10: Proceedings of the 19th international conference on World wide web. New York, NY, USA: ACM, 2010, pp. 811–820.
4. A two-stage ensemble of diverse models for advertisement ranking in KDD Cup 2012[C]//KDDCup. 2012.
5. Jahrer M, Toscher A, Lee J Y, et al. Ensemble of collaborative filtering and feature engineered models for click through rate prediction[C]//KDDCup Workshop. 2012.
