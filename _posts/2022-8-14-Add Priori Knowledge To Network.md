---
layout: article
title: [神经网络加入先验]记录
mathjax: true
tags: Papers
---

# [神经网络加入先验]记录

* **人对数据的理解**作用于**特征工程**的时候，会提升模型

* **数据和特征决定了机器学习的上限，模型和算法只是逼近这个上限而已**

## 1 数据、特征层次

* **特征工程**不是传统机器学习算法的专属

* 神经网络在非线性建模上有着无可比拟的优势，但一些**特定的任务**还是非常考验人的先验的
  
  * 声纹识别领域，必须借助高级的特征工程，例如MFCC，这里面包含了大量的人工步骤和经验参数，预加重，分帧，加窗，快速傅里叶变换(FFT)，梅尔滤波器组，离散余弦变换(DCT)

* **高阶的特征(专家知识)** 尽量放在NN后面（不严谨）

* <推荐系统问题中特征加入>

![2.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/14-18-09-39-2.png)

* 特征加在[Wide & Deep](https://zhuanlan.zhihu.com/p/47293765)?
  
  ![3.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/14-18-11-46-3.png)

* 重要特征作为裁判
  
  ![3.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/14-18-15-45-3.png)
  
  

## 2 模型结构层次

人对数据和任务理解，在网络设计中具体的表达形式，初衷都是**人的先验在神经网络中学习组件的具象化**：

* **图像分类常用的CNN中使用的卷积层**：局部假设，利用了图像像素距离越近相关性越强的先验

* **NLP中self-attention**：邻近token相关性

* 

## 3 网络参数层次

* 将知识融入到网络参数中，使用了很多**目标任务选择**方式的技能
  
  * **NLP**里从传统神经语言模型，发展到**完形填空**
  
  * **CV**里从大数据集分类预训练到**图像复原重建**

* **知识蒸馏**：
  
  ![1.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/14-17-53-01-1.png)
  
  

## 4 目标约束层次

* **对比学习：** 典型的在目标约束层次+数据和特征层次引入先验的
  
  ![4.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/14-18-26-05-4.png)

* * 单/多目标单调性约束
  
  * 层次正则化约束：包括各种先验的正则化，通过目标层级无环图，设计分层L2正则，约束了链接关系相近的类别，其预测值也要接近。



> [1] [NLP->CV预训练](https://mp.weixin.qq.com/s?__biz=MzIwNDY1NTU5Mg==&mid=2247484129&idx=1&sn=757ca5f4ef611eb95b6e6d748823fb71&chksm=973d9c66a04a1570c630a846cabd8b2d5e24ae9d67ca87f4562fdd66808530e6978f675a1d26&scene=21#wechat_redirect)
> 
> [2] [神经网络加入先验](https://zhuanlan.zhihu.com/p/456795337)
> 
> [3] [专家知识编码入神经网络](https://www.zhihu.com/question/529959915/answer/2470689232)


