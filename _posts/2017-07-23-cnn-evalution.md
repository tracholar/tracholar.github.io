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

* 目录
{:toc}

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
另外的一些网络结构则将传统的卷积层的两个功能——空间卷积和特征混合分离出来，形成了1x1卷积层和Deepwise Separate Convolution层的两层结构来替代原有的单卷积层。

### 标准的卷积层(Convolution Layer)
相比于全连接的神经网络，卷积网络的一大改进是采用局部连接并且在空间上共享权值——即卷积操作。
局部连接的对于图像、语音、词序列等具有局部相关性或具有局部模式的信号非常重要，局部连接可以很好地提取出这些局部的特征模式。
这种局部连接的想法受到了生物学里面的视觉系统结构的启发，视觉皮层的神经元就是局部接受信息的，这些神经元只响应某些特定区域的刺激。另一方面，自然图像固有的空间平稳统计特性使得在不同的区域使用相同的模板去抽取相同的特征，这就是权值共享。
下图是[UFLDL](http://deeplearning.stanford.edu/wiki/index.php/%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)上的一个简单的例子，形象地解释空间卷积操作的过程。

![空间卷积](/assets/images/conv2d.gif)

上图显示的是对单通道图片的单个卷积核的卷积过程，对于边界直接截断，所以卷积后的尺寸比原始图像尺寸要小一些。
一般地，对于多个通道的图像，如自然图像具有三个颜色通道，利用多个卷积得到的多个 feature map 构成的多通道，
卷积核除了要对空间维度进行卷积操作外，还会在通道维度进行全连接变换。假设输入图像用$$X_{ijf}$$表示，
其中i,j代表空间维度的下标，f代表通道；用$$w_{klf}$$代表一个卷积核，其中k,l代表空间维度，f代表通道，
那么经过卷积后输出的图像为

$$
Y_{ij} = \sum_f \sum_l \sum_k X_{i-k, j-l, f} w_{k, l, f} .
$$

可以看到，多通道卷积操作具有两个功能，一是对空间维度进行卷积，而是混合输入图像的不同通道，这些通道可以是颜色通道，也可以是特征通道，因此相当与在特征维度进行全连接！后面可以看到，一些新的卷积层将这两个功能进行分离！
卷积层中有一种特殊的卷积核——1x1卷积，它的作用相当于只实现了特征维度的全连接操作，而没有空间维度的卷积操作，广泛地应用于特征降维、局部区域分类等应用当中，后面我们会经常碰到。

![多通道卷积](/assets/images/Conv_layer.png)

卷积操作还可以引入 stride，每次移动卷积核的步长可以大于1，可以减少输出的空间维度实现降维。
另一个角度来看，它相当于将下采样的功能集中到卷积层当中，在卷积层中实现了Pooling操作。


在卷积网络中，卷积之后的结构往往会进行非线性操作后才作为该层的输出，$$f = \sigma(Y + b)$$，$$\sigma$$是非线性激活函数，早期用得比较多的是 sigmoid，tanh激活函数，现在基本上都用 ReLU 激活函数及其变种，如 leaky ReLU[12]，以及最近的研究提出自归一激活函数 SELU[13]。非线性激活函数的提供了整个网络的非线性变换特性。另一个提供非线性变换的地方是下面的池化层——Pooling。

每一个卷积核可以提取一个特定模式，所以一般的卷积层都会有多个卷积核，用于提取多个局部模式。

### 池化层(Pooling Layer)
池化操作是一种降维的方法，它将一个小区域通过下采样、平均或者区最大值等方式变成单个像素，实现降维。
池化操作除了降维外，还能在一定程度上实现局部平移不变性，即如果图片朝某个方向移动一小步，如果利用Max Pooling，
那么输出是不变的，这使得分类器具有一定的鲁棒性。因为这个原因，Max Pooling操作在卷积层中应用最为广泛。
下图是 Pooling 操作的示意图，来自 UFLDL[14]。而 Avarage Pooling 常用在全连接层的前面，将高层的卷积层输出的特征在空间维度平均，送入一个线性分类器当中。

![池化操作](/assets/images/Pooling_schematic.gif)

与卷积层类似，Pooling操作也可以引入stride，即跳过特定的距离进行 Pooling。上述示意图实际上是stride=Pooling的大小的特定情况。

### 小卷积核

### Bottneck结构

### Inception结构

### Deepwise Seperator Convolution

### grouped Convolution

### 残差连接模式



## 从LeNet到ResNet

## 从AlexNet到ShuffleNet

## 从ZFNet到DeepDream

## 总结与展望

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
12. Xu B, Wang N, Chen T, et al. Empirical evaluation of rectified activations in convolutional network[J]. arXiv preprint arXiv:1505.00853, 2015.
13. Klambauer G, Unterthiner T, Mayr A, et al. Self-Normalizing Neural Networks[J]. arXiv preprint arXiv:1706.02515, 2017.
14. Andrew Ng 的UFLDL教程 <http://deeplearning.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B>
