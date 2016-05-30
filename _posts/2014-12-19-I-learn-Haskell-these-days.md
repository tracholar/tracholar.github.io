---
layout: post
title: "我学习Haskell的这些天"
description: ""
category: "techology"
tags: ["haskell","programming","functional programming"]
published: false
---

最近一段时间，一直在学习一门叫Haskell的编程语言。
这门编程语言与之前学习的C、Java、Javascript、Python等语言完全不同，简直快颠覆了我编程的三观了。


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


