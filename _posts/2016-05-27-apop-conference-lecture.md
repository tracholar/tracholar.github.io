---
layout: post
title: "APOP conference lecture"
description: ""
category: "lecture"
tags: ["学术会议","量子接收机"]
published: false
---

Oral presentation text article.

# Slide 1
Good morning, everyone! I’m Yuan Zuo.The report I present today is named “ six-teen QAM quantum receiver with hybrid structure outperforming the standard quantum limit”.

# Slide 2
This is the outline. My lecture consists of five parts. First I will give a brief introduction about the background. Then I will show the receiver structure and the strategy for six-teen-QAM signals. After that, I will give a theoretical analysis of the receiver performance and numerical simulation results. And in the final part, some conclusions of our work are give. OK, let me show you the details.

# Slide 3
First of all, let me introduce some basic concept. In classical optical communication system, the error probability is limited by shot noise. System without thermal noise is modeled by addictive Gaussian white noise model. It is show in this picture. The ultimate limit for this system is called standard quantum limit or SQL for shot. The ultimate limit means that for the given pow, the error probabilities is limited by SQL.
In quantum detection and estimation theory, the ultimate limit is Helstrom limit.
Which is lower than the SQL. These theory is proved by Helstrom in 19sixtys. In this theory, the optimal measurements are described by a set of positive operate. And we can solve this optimization problem to work them out.
Unfortunately, it is hard to design a receiver using optical devices to achieve the Helstrom limit.
So far, some receiver have been proposed to approach the ultimate limit.
Such as Kennedy receiver, Dolinar receiver, Bondurant receiver and so on.

# Slide 4
OK, I have introduced some basic concept. Standard quantum limit SQL and Helstrom limit. So what’s my job? We design a receiver for six-teen Quadrature Amplitude Modulation signals. These modulation is also abbreviated as QAM signals. The performance of this receiver is lower than the SQL.
Why do we do that? There are two key reasons. Firstly, some quantum receivers for binary modulation, phase shift key modulation and pulse position modulation signals have been proposed, but few for QAM signals. Secondly, QAM signals have a higher spectral efficiency than other modulations. So it have been used in high capacity optical communication and wireless communication. If a receiver could reduce the error rate, the capacity will be further improved.
How to design this receiver? We use two skills. Hybrid structure and optimal displacement strategy. I will describe my idea in detail in the following slides.

# Slide 5
In order to discriminate the sixteen-QAM signals below the SQL, we designed a hybrid structure scheme. The receiver structure diagram is shown in this picture. It consists of a homodyne detector and a displacement receiver. At first, signals are split into two part. The first beam is feed into the homodyne receiver. And the homodyne detector measure the P quadrature value. The results are feed forward to the displacement receiver to measure the X quadrature value. The correct detection probabilities are given by the product of two probabilities. It equals P HD tiems P DR.   The HD stands for the homodyne and DR stands for the displacement receiver.

# Slide 6
OK, let me show you the process step by step. At first, incoming signal is plit into two portions by a beam splitter (abbreviate as BS). The homodyne detector measures the first beam to discriminate the P quadrature value. It is described by these four positive operator value measurement mathematically. The domain of integration of these four measurements are different. They are shown in this picture. So, we can calculate the correct probabilities of the homodyne detector by sum these four formula and divided by four.

# Slide 7
The next step is to measure the X quadrature value. Suppose the homodyne yield correct results, then six-teen hypothesis [haɪ'pɑθəsɪs] are reduced to only four hypothesis. The four signals are discriminate by a displacement receiver – Bondurant receiver. The receiver strategy are shown in the right side. At first, it displace the first signal into vacuum ['vækjum] state. The displacement operator can be implemented by a Mach-Zehnder [mɑk  zender] interferometer [ˌɪntəfɪ'rɒmətə].  The displaced optical field is detected by a single photon detector. If we guess right, there are no photons will be detected. But if we are wrong, the single photon detector is very likely to catch a photon. Then we guess the next one and change the displacement operator to displace the next signal to vacuum. These steps are repeated when we got the second photon and the third photon. They are repeated until the end of the signal period.

# Slide 8
In the second step, the four probabilities are calculated by following four integration. And we sum them and divide by four to get the correct probabilities of the displacement receiver.

# Slide 9
In the previous slide, we displace the signal into vacuum state. It is exact nulling operator. However, some literatures report the exact nulling strategy is not good idea for weak signals. The optimal displacement strategy is more efficient. This strategy not displace the signal into vacuum but an additional displacement beta. Like this. The value of beta is optimized by numerical method.

# Slide 10
We simulate our receiver under two different configurations. Exact nulling and optimal displacement. The left figure shows the additional displacement over signal average photon number. The additional displacement decays [di’kei] with the photon number. The right figure show the performance of the receivers. The black line stands for Helstrom limit, and the red line stands for the standard quantum limit – SQL. The blue line stands for our receiver with exact nulling strategy. And the green dash line stands for the optimal displacement strategy. When the photon number is big enough, our two receiver can both outperform the SQL, and their performance are almost coincident. At the weak signal region, the optimal displacement strategy is better than the exact nulling strategy.

# Slide 11
OK, in this lecture, I have report two type quantum receivers for six-teen QAM signals. And the numerical simulation shows that these two receivers can both outperform the SQL. And the optimal displacement strategy can reduce the error probabilities, especially when signals is weak. Since these two receivers can reduce the error probabilities, it is potential to apply them to increase the distance and capacity in optical communication.

# Slide 12
Thank you!
