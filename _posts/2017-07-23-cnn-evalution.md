---
layout: post
title: "卷积网络的进化史"
description: ""
category: "machine-learning"
tags: ["卷积网络","机器学习","深度学习"]
---

## 前言
学习卷积网络有一段时间了，对卷积网络的演变过程中的一些基本思路有一个大致的理解，
于是总结出了这篇文章，一方面是加深自己的理解，一方面也希望对学习卷积网络的读者有所帮助。

## 综述
卷积网络是一种利用卷积提取序列或者空间数据局部或全局模式的一种网络，
其核心部分是所谓的卷积。事实上，卷积是一种特殊的数学运算，最早追溯到线性系统的建模。
线性是不变系统可以用一个冲击响应完备地描述，对于任何一个输入的信号，都可以将输出信号表达为输入信号与该系统
的单位冲击响应的卷积！从网络连接角度来看，卷积相当于局部连接并且在不同时间或者空间上共享连接权重。
一个卷积核事实上就是在提取某一种特殊的模式，事实上在图像处理系统中，很早就利用各种卷积核对图像进行处理，
例如高斯模糊核、拉普拉斯核、solber核等等。

最早将卷积操作引入神经网络中的工作应该是 LeCun 在1998年提出的 LeNet-5 [1]，那里用到了现代卷积网络的两种基本操作：
卷积和池化pool，不同的是，Pooling用的是下采样的方式，而不是现代主流的 Max Pooling 或者 Avg Pooling。
但是，此后的10几年，由于计算能力的不足和标准数据的缺乏，卷积网络的效果一直不如浅层的网络。直到2012年，Hinton的学生 Alex Krizhevsky 利用一个8层的卷积网络，一举夺下著名的 ImageNet 比赛冠军，才让人们重新关注起神经网络[2]。可以说，Alex提出的AlexNet以及DBN在语音识别的重大突破[3]，成为现在火热的深度学习爆发的导火索。因为，人们看到，在当时无法理解的所谓深度神经网络的帮助下，图像识别的准确率遥遥领先传统的方法10个百分点，让十年没啥突破的语音识别技术提升了9个百分点！于是，大家虽然不明白里面到底发生了什么，但是知道，这应该是未来最重要的方向没有之一。

自AlexNet后，一批批学者开始探索卷积网络的不同结构的影响，以及对卷积网络的可视化分析，让我们对卷积网络的认识越来越直观与深入。
其后的几个代表性工作如 ZFNet，VGGNet，GoogLeNet，Inception各种版本，ResNet etc[4-9]。探索了对卷积核的该进、连接方式的改进以及对卷积层的改进。使得模型能够越来越深，参数和计算复杂度越来越低，但是性能越来越好。理论上来讲，残差网络（ResNet）已经构造出一种结构，可以不断地增加网络的层数来提高模型的准确率，虽然会使得计算复杂度越来越高。它证实了我们可以很好地优化任意深度的网络。要知道，在那之前，在网络层数达到一定深度后，在增加反而会使得模型的效果下降！因此，残差网络将人们对深度的探索基本上下了一个定论，没有最深，只有更深，就看你资源够不够！另一方面，如何优化网络结构，使得较小的网络和较浅的网络也能做到极深网络的性能，但是计算复杂度和参数数目都控制在较小的范围内。这在移动端、嵌入式系统中的部署十分关键！在这方面，也有一些代表性工作，如 MobileNet [10]，ShuffleNet [11]。未来，设计更高效的网络将是研究者的重要研究方向之一，其中高效应该包括用更少的参数、更少的计算资源、以及更少的标注样本！

## 基本组件
卷积网络一般都会包括两个基本组件：Convolution Layer, Pooling Layer.
可以说，是否包含这两个基本组件是区分卷积网络与其他类型的网络的重要标准。
一些新的网络结构通过使用大于1的stride，来实现空间维度的下采样降维，这种方法可以让网络结构中不显示地包含 Pooling 层，
可以理解为将 Pooling 的功能融合到卷积层当中。
另外的一些网络结构则将传统的卷积层的两个功能——空间卷积和特征混合分离出来，行程了1x1卷积层和Deepwise Separate Convolution层的两层结构来替代原有的单卷积层。

### 标准的卷积层


## 从LeNet到ResNet

## 从AlexNet到ShuffleNet

## 总结

## 参考文献
1. LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324. <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>
2. Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.
3. Hinton G, Deng L, Yu D, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups[J]. IEEE Signal Processing Magazine, 2012, 29(6): 82-97.
4. Zeiler M D, Fergus R. Visualizing and understanding convolutional networks[C]//European conference on computer vision. Springer, Cham, 2014: 818-833.
5. Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
6. Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1-9.
7. Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning[C]//AAAI. 2017: 4278-4284.
8. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
9. Chollet F. Xception: Deep Learning with Depthwise Separable Convolutions[J]. arXiv preprint arXiv:1610.02357, 2016.
10. Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv preprint arXiv:1704.04861, 2017.
11. Zhang X, Zhou X, Lin M, et al. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices[J]. arXiv preprint arXiv:1707.01083, 2017.
12.
