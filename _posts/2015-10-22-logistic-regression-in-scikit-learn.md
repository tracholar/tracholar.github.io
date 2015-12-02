---
layout: post
title: "sklearn中的logistic回归实现"
description: ""
category: "techology"
tags: ["python","scikit-learn","源码"]
---

Logistic回归在机器学习的分类问题中占了非常重要的地位，
因为它简单可解释，容易实现，训练速度快，不存在局部最优解，
容易并行等特点，让它成为业界应用最为广泛的算法。
scikit-learn是著名的机器学习算法库，用python实现，文档丰富，
在很多机器学习相关的比赛(如kaggle)中，被推荐使用。
今天就来看看scikit-learn中，logistic回归是如何实现的。
