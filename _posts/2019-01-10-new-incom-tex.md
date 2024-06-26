---
layout: post
title: "理解新个税的累计预扣法"
description: ""
category: "math"
tags: ["个税","数学"]
published: false
---

* 目录
{:toc}


今天刚发了12月份的工资，看了一下工资条发现，我擦嘞！扣税金额怎么比上个月少了不少，这帮HR是不是算错了，税务局会不会找我麻烦！后来才知道，原来是实行了新个税的「累计预扣法」。那么，累计预扣法和按月扣差别大吗？然而看了一眼HR姐姐发的科普文章，一脸懵逼。

![啥啥啥](/assets/images/sha-sha-sha.jpg)

不过，作为好奇宝宝，我还是通过查阅资料，演算了一摞草稿纸，总算搞明白了这个奇葩的扣税方法。结论是，**如果你每月边际税率(就是指你最高的那一级税率)是相同的，那么对你全年税后总收入没影响，否则，你就占了点国家的便宜**。

解释一下**边际税率**，比如你扣除了各种免征额后的应税收入是4000元，那么其中3000元按照3%的税率扣税，而剩下的1000元按照10%的税率扣税，最高税率10%就叫边际税率。它意义是，在现有收入的基础上，每多给你发1元，就要扣掉其中10%的税。

| 级数 | 按月应纳税所得额     | 税率（%） |
| ---- | ---------------------------- | --------- |
| 1    | 不超过3000元的         | 5         |
| 2    | 超过3000元至12000元的部分 | 10        |
| 3    | 超过12000元至25000元的部分 | 20        |
| 4    | 超过25000元至350000元的部分 | 25        |
| 5    | 超过35000元至55000元的部分     | 30        |
| 6    | 超过55000元至80000元的部分     | 35        |
| 7    | 超过80000元部分               | 45        |


## 按月扣除的缺点
在以前，每月收入都是单独扣税，当月扣税金额跟之前的没有关系，这就会导致如果你某个月收入特别高(比如发了一笔巨额奖金)，而被扣了很高的税。举例而言，假设你一年11个月的应税收入是1000(再次强调一下「应税收入」或者「应纳税所得额」是指扣除所有免征额后的金额，包括五险一金、5000起征点、专项扣除等等)，这11个月你的税率都是3%。而最后这一个月多发了24000元奖金，那么这个月应税收入就是25000元，除了前面3000元的税率是3%，后面21000元的税率都超过了3%，而且还有一部分的税率达到了20%(参见下图的平滑之前)。在这种情况下，你全年应税收入36000元，其中14000元扣了3%的税，9000元扣了10%的税，剩下的13000元扣了20%的税！！

![平滑前后](/assets/images/tax-compare.png)

相反，如果这多出来的24000元奖金可以平均分摊到你的12个月里面，那么你每个月应税收入恰好是3000元，你全年36000元的应税收入都只要按照3%的税率扣税！参考上图，平滑之前你全年应税收入中有一部分是10%的税率(蓝色)，还有一部分是20%的税率(红色)。而平滑之后所有应税收入都按照3%交税了(黄色)。所以，在不增加全年的应税收入的情况下，平滑每月收入可以帮你少扣不少的个税(见下表)。

| 税率 | 平滑前应税收入部分 | 平滑前扣税金额 | 平滑后应税收入部分 | 平滑后扣税金额 |
| ---- | ------------------ | -------------- | ------------------ | -------------- |
| 3%   | 14000              | 420            | 36000              | 1080           |
| 10%  | 9000               | 900            |                    |                |
| 20%  | 13000              | 2600           |                    |                |
| 总计 | 36000              | 3920           | 36000              | 1080           |

很明显，这两种扣税方法的差异只有在按月扣除时，每个月的边际税率不相同的时候发生。如果每月的边际税率都相同，那么这两种方式并没有差异。这可以从全年总金额的角度来考虑，参考下图，把全年总的应税收入按照税率分组(图中用不同颜色表示不同的税率)，如果每月边际税率相同，那么除了边际税率那一组(蓝色那组)之外，其他组(黄色那组)两种方式每月的扣税金额都一样。而对于边际税率那一组，两种方式每月扣税金额不一样，但是总金额是相同的。所以，全年总的扣税金额也相同。

![平滑前后](/assets/images/tax-compare2.png)


## 累积预扣法

上述案例说明，要想扣税少，每月收入平均扣税最少。累积预扣法则是从第一个月开始，对你的应税收入进行累积，前面36000元应税收入全部按照3%扣税，扣完这部分之后，超过36000不超过144000的按照10%的税率扣除。这样就不用平滑，也能做到和平滑相同的扣税效果。唯一的区别是，最后那几个月的税率比较高(参考下图累计预扣法部分)。

![累计预扣法](/assets/images/cum-tax.png)

因此，累积预扣是按照全年的角度来扣税的，按照你的应税收入收到的时间顺序，前36000按照3%扣税，接下来36000-144000按照10%扣税，以此类推(参考下表)。因此，这会导致在年初的时候，扣税比较少，到手收入多，然而到了年底，扣税就会比较多，收入偏少。但是，从全年总体的角度来说，到手收入不会比之前按月扣除的少！如果收入波动较大，比如销售、中介这些职业，反而到手收入会变多！这也算是国家给我们发的福利吧。

| 级数 | 全年应纳税所得额     | 税率（%） |
| ---- | ---------------------------- | --------- |
| 1    | 不超过36000元的         | 3         |
| 2    | 超过36000元至144000元的部分 | 10        |
| 3    | 超过144000元至300000元的部分 | 20        |
| 4    | 超过300000元至420000元的部分 | 25        |
| 5    | 超过420000元至660000元的部分 | 30        |
| 6    | 超过660000元至960000元的部分 | 35        |
| 7    | 超过960000元的部分     | 45        |

## 累计预扣相比按月扣除有利场景

1. 收入波动大的职业，比如销售、中介主要靠提成和奖金，某些房产中介卖出一套豪宅佣金就有几十万，如果按月扣税，当月得扣掉近45%的收入，但是按照累计预扣法则相当于将这几十万摊到全年扣税，总体上扣税就少了很多。以20W为例，按月扣除大约扣除近9W的税，而均摊后，则只有2W左右的事税了！！
2. 年底双薪的企业，年底双薪通常会导致税率跨级，按照累积扣除则可以少扣一些个税。
3. 一年中只有部分月份工作的人
4. 上半年上班，下半年休假的佛系员工
5. 刚入职的高收入应届生在第一年的时候，可以将半年的收入摊到全年扣税，可以少扣不少个税
