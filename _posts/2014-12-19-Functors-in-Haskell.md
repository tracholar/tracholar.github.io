---
layout: post
title: "Haskell中的涵子(Functors)"
description: ""
category: "techology"
tags: ["haskell","programming","functional programming","Functors"]
---

## 涵子 Functor
涵子是一种特殊的对象（对，没错，是对象而不是函数），他可以被map over，比如lists, Maybe, Tree等。
它的定义是
`
class Functor f where
  fmap :: (a -> b) -> f a -> f b
`
`fmap`函数可以认为是传入一个作用于普通类型的函数，而传回一个作用于一个涵子类型的新函数。
以list int为例，list的`fmap=map`，我们知道对于list int类型，`map`是这样一个函数，
他传入一个作用于`int`类型的函数，而传回一个作用于`list int`类型的函数。

### 第一定律
`fmap id = id`

### 第二定律
f和g是任意两个可以复合的函数，即`f.g`是一个合法的函数。
那么，`fmap`应该满足`fmap (f.g) = fmap f . fmap g`。
