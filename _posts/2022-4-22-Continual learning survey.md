# A continual learning survey: Defying forgetting in classification tasks

> - **Publisher:** [ T-PAMI ]
> 
> - **Date of Publication:** 16 Apr 2021
> 
> - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Machine Learning (stat.ML)
> 
> - **论文链接：**[[1909.08383] A continual learning survey: Defying forgetting in classification tasks](https://arxiv.org/abs/1909.08383)
> 
> - **Data of Read**: 2022.4.23
> 
> - **研究问题:** 增量学习综述（很厉害！）

# Abstract

- **持续学习的目标就是将这种模式转变为可以从不同任务中不断累积知识而不需要从头再训练的网络。**

- **贡献：**
  
  * 对最先进技术的分类和广泛概述
  
  * 提出了一个新的框架来确定continual learner的稳定性和可塑性的权衡。
  
  * 对11中最先进的持续学习方法和4种基线进行了全面的实验比较

- 我们研究了**模型容量、权重衰减和dropout正则化**的影响，以及**任务呈现的顺序**，并从所需内存、计算时间和存储容量等方面定性地比较了各种方法。

# 1. Introduction

## 1.1 Background

### 1.1.1 持续学习

- 研究从**无限的数据流**中学习的问题，目标是逐步拓展已获得的知识，并用于未来的学习中

- 数据可以从**不同的域**获取（例如不同的成像条件），也可以与**不同的任务**相关联（如细粒度的分类问题）

- **标准**：学习过程的**顺序性**，一次只能从一个或几个任务中获得一小部分输入数据

- **挑战**：不发生**灾难性遗忘**的情况下学习，即以前学习的任务或领域的性能并不能因为随着新任务或新领域数据的学习而发生显著性下降

- **稳定性-适应性困境**：适应性指整合新知识的能力，稳定性是指编码时保存持原有知识的能力

## 1.2 Motivation

* 研究持续学习任务要使用的**实验设置**和**数据集**很少有统一的共识，尽管论文中提供了在特定的模型设置下提出方法的优越性，但至今没有全面的实验对比

## 1.3 论文内容

* 对**最新技术**进行**分类**和广泛**概述**

* 为了进行公平的方法比较，提出**一个框架**来解决**超参数的选择**问题

* 对**11种**最先进的持续方法和**4个baseline**进行综合实验比较

# 2. The Task Incremental Learning Setting（任务增量学习设置）

        由于持续学习的普遍困难和挑战的多样性，许多方法将一般设置放松为更简单的任务增量设置。设置一系列任务，每次只接受一个任务的训练数据进行训练。

- 目标是控制所有已知任务的**统计风险**：
  
  <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/24-18-18-17-111.png" alt="111.png" width="361" data-align="center">

- 对当前任务T，统计风险可以近似为**经验风险**
  
  <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/24-18-18-33-222.png" alt="222.png" width="353" data-align="center">

> [3级] 增量学习与持续学习的关系与区别？

        

# 3. Continual Learning Approaches（持续学习方法）

        下图以树结构展示了持续学习的几个分支：**重放方法(Replay methods)、基于正则化的方法(Regularization-based methods)、参数隔离方法(Parameter isolation methods)**，这几个分支下面还会进行分流，然后再到具体的实现算法。

<img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/24-17-59-50-1.png" alt="1png" width="519" data-align="center">

## 3.1 **持续学习方法**

* **重放方法**
  
  >        **存储原始样本**，或使用生成模型**生成伪样本**。在学习新任务时，这些之前任务的样本会被重复播放：
  > 
  >         a) 被重新作为**模型训练输入**
  > 
  >         b) 约束新任务的**优化损失**（防止之前的任务干扰）

* **基于正则化的方法**
  
  >         避免了存储原始输入、优先考虑隐私和减轻内存需求。在损失函数中引入了一个额外的**正则化项**。划分为以**数据为中心**的方法(Data-focused
  > methods)和以**先验为中心**(Prior-focused
  > methods)的方法。

* **参数隔离的方法**
  
  >         为每个任务指定不同的模型参数，以防止任何可能的遗忘。

# 4. CONTINUAL HYPERPARAMETER FRAMEWORK（持续超参数框架 ——  解决超参数的选择问题）

>         解决持续学习问题的方法通常涉及**额外的超参数****，以平衡**稳定性和适应性之间的权衡**。
> 
>         这些超参数通常是通过**网格搜索**（通过遍历给定的参数组合来优化模型）找到的，需要使用所有任务中保留的验证数据，违反了持续学习的设置。

作者提出的框架主要包含两个阶段：

* **最大适应性搜索**
  
  * 在原模型上使用新任务数据进行finetune
  
  * 通过粗网格搜索得到“超参数”（学习率...），使得在新任务上获得最高精度

* **稳定型衰减**
  
  * 与**遗忘**有关的超参数**H**设置为最高值，保证遗忘率最小
  
  * 用阶段一获得的“超参数”训练网络
  
  * 定义**阈值p**：表示新任务性能的容许下降程度
    
    * 当不满足该阈值时，通过将**衰减因子与H相乘，减小H值**（即加大网络对之前学习结果的遗忘），重复该阶段
    
    * 相当于**增加模型的适应性**，以达到所需的性能阈值

* **算法流程**
  
  <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-19-38-40-1.png" alt="1.png" data-align="center" width="287">

# 5. COMPARED METHODS（持续学习方法比较）

* **重放方法**（Replay methods）
  
  * **iCaRL**: 
    
    * 存储最接近每个类的特征平均值的样本
    
    * 训练时: 最小化对新类的估计损失以及以及先前模型预测的目标以及当前模型对先前学习的类之间的蒸馏损失。
    
    * **框架中的H**：蒸馏损失强度（由于蒸馏损失强度与保留之前的知识相关）
  
  * **GEM**:
    
    * 通过一阶泰勒级数近似将估计（通过previous task data buffer 中选择的样本来估计）的梯度方向**投射到先前任务梯度所描述的可行区域上**来实现。
    
    * **框架中的H**:用来**改变梯度投影**的一个小常数**γ**

* **基于正则化的方法**（Regularization-based methods）
  
  1）基于**数据**的方法：
  
  * **LwF**： 在训练新任务之前，记录寻人我数据的网络输出作为软标签，将知识从以前任务中得到的模型蒸馏到根据新数据训练的模型
  
  * **EBLL**：通过保留以前任务的重要低维特征，扩展LwF。对每项任务，一个未完成的自动编码器将特征投影到低维流形，将任务特征约束在相应的低维学习空间
  
  2）基于**先验**的方法：
  
  * **EWC**：估计所有神经网络的参数重要性，在后期任务训练中，对**重要参数改变进行惩罚**
    
    * 重要性权重：
      
      <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-20-01-12-1.png" title="" alt="1.png" data-align="center">
  
  * **SI**：在训练中保持在线估计w<mark>(?)</mark>
    
    * 重要性权重：
      
      <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-20-01-23-2.png" title="" alt="2.png" data-align="center">
  
  * **MAS**：无监督
    
    * 重要性权重：
      
      <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-20-01-39-3.png" title="" alt="3.png" data-align="center">
  
  * **IMM**：
    
    * Mean-IMM:
      
      <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-20-01-59-4.png" title="" alt="4.png" data-align="center">
    
    * MODE-IMM:
      
      <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-20-02-13-5.png" title="" alt="5.png" data-align="center">

* **参数隔离方法**（Parameter isolation method）
  
  * **PackNet**
    
    * 通过构造二进制掩码，迭代地为连续任务分配参数子集。使用模型剪枝的方法得到对之前任务重要的参数，使用剩余的参数训练新增任务。
    * H：包含每层修剪分数。
  
  * **HAT**
    
    * 对每层网络嵌入通过Sigmoig进行注意力掩膜，训练中校正Sigmoid的斜率。
    
    * 正则化项在新任务注意掩膜上施加稀疏性，在两个被认为对以前的任务很重要的单元之间限制参数更新。
    
    * H：同时考虑了正则化强度和Sigmoid斜率。

# 6. Experiments（实验）

## 6.1 基本介绍

### 6.1.1 数据集

* **Tiny Imagenet**  ：200个类别，每个类别500个样本。我们在10个连续任务的序列中为每个任务分配20个随机选择的类。所有任务在难度、大小和分布方面大致相似
  
  <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-19-01-13-1.png" title="" alt="1.png" data-align="center">

* **iNaturalist**：提供一个更真实的环境，其中包含大量细粒度类别和高度不平衡的类。从14个超级类别中选择了最平衡的10个，并且只保留了至少100个样本的类别
  
  <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-19-01-26-2.png" title="" alt="2.png" data-align="center">

* **RecogSeq**：由8个连续数据集组成
  
  <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-19-01-40-3.png" title="" alt="3.png" data-align="center">

### 6.1.2 模型

* Tiny Imagenet：用VGG作为基础网络，通过改变模型的深度（即不同Block的数量）得到不同的**模型容量**：SMALL、BASE、WIDE、DEEP

![1.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/22-19-02-27-1.png)

* 对iNaturalist/RecogSeq：用AlexNet作为基础网络

### 6.1.3 评价指标：

* **准确率**

* **遗忘率**：首次学习任务时的准确率与训练一个或多个额外任务后准确率的差异

（后面实验部分将围绕这两点展开对比）

### 6.1.4 Baseline

* **finetuning（微调）**:不考虑之前的任务性能，导致灾难性遗忘，代表最低期望性能

* **joint（联合训练）**：同时使用任务序列中的所有数据，违反了持续学习设置，代表最高期望性能

* 针对**重放的方法**：
  
  * **R-FM**:充分利用总的可用记忆,所有之前的任务等分内存容量。有内存管理策略
  
  * **R-PM**:所有任务上预先分配固定样本记忆。任务数量T预先知道。没有内存管理策略

### 6.1.5 重放缓冲区

* **过大的缓冲区**将导致不公平比较，甚至在其极限内保存以前任务的所有数据，如联合学习，这不符合持续学习设置。因此，我们使用存储基本（BASE）模型所需的内存

* 默认4.5k缓冲区,并对9K缓冲区进行了实验

## 6.2 方法比较及部分影响因素

### 6.2.1 在**Base模型**上比较所有方法

![图片1.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-09-55-38-%E5%9B%BE%E7%89%871.png)

* Tiny imagenet数据集上，任务随机排序

* PackNet新任务上的性能通常更差。然而，它允许完整的知识保留，直到任务序列结束（图中没有遗忘）。

* 由于基于最近邻的分类，iCaRL一开始对新任务的准确率要低得多，但随着时间的推移，它会有所提高。放大缓冲区效果更好。

* MAS对超参数值的选择更加稳健，高于EWC和SI（可能过拟合）。

* 在任务1到7中，R-FM方案比R-PM的效果要好得多，但在最后三个任务中没有。当看到更多任务时，两条baseline收敛到同一个内存方案，对于最终任务，R-FM为序列中的每个任务分配相等的内存部分，因此相当于R-PM。

### 6.2.2 模型容量影响

![图片2.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-09-57-13-%E5%9B%BE%E7%89%872.png)

![图片3.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-09-57-19-%E5%9B%BE%E7%89%873.png)

* SI很容易出现过度拟合问题（可能会通过正则化克服），而PackNet在一定程度上防止了过拟合

* iCaRL 9k和PackNet仍保持领先

* 小型模型的精确度不高，主要由于遗忘严重（紫框）

* 深层模型的联合训练效果较差（蓝框）。这已经表明深度模型可能不是最优的

* 由于网络容量的降低，微调和SI会发生严重的遗忘（红色下划线）

* 宽模型中SI好于EWC

* 一些模型受益于小网络（绿色下划线）

* 使用额外的卷积层扩展基础模型会导致性能降低（黄框）

### 6.2.3 正则化影响

研究了两种正则：**dropout**和**weight decay**

![图片4.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-09-59-52-%E5%9B%BE%E7%89%874.png)

<img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-09-59-59-%E5%9B%BE%E7%89%875.png" title="" alt="图片5.png" data-align="center">

* 负面结果用红色下划线

* Packnet效果最好; HAT dropout导致更差的结果，可能是因为干扰了基于单元的掩膜; SI效果最显著; EWC和MAS受到额外正则化的干扰。（重要参数被丢弃或衰减）

* 重放baseline几乎所有模型受益于dropout,但影响有限。

### 6.2.4 真实环境（数据分布）

影响使用不平衡的任务序列

<img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-10-02-05-%E5%9B%BE%E7%89%876.png" title="" alt="图片6.png" width="613">

![图片7.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-10-02-09-%E5%9B%BE%E7%89%877.png)

* iNaturalist：
  
  * Packnet平均准确率最高
  
  * Mode-IMM表现出反向转移
  
  * 学习任务5时，使用基于先验的方法EWC、SI和MAS会出现显著的下降，可能因为只包含5个类和很少的数据，强制网络过度适应这项任务
  
  * EBLL限制了新任务特征与之前任务特征的最佳呈现方式密切相关，使平均准确度比LwF显著提高7.91%

* RecogSeq:
  
  * 在学习最后一个SVHN任务时，会出现强烈的遗忘
  
  * PackNet以高正确率领先，接近联合表现

### 6.2.5 任务顺序影响

![图片8.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-10-04-19-%E5%9B%BE%E7%89%878.png)

* Tiny Imagenet:
  
  * 通过测量在每个任务的数据集上获得的4个模型的准确性得到**从难到易**的排序
  
  * 我们希望简单到困难的排序能够提高精度（ PackNet和MAS ），而随机排序则相反。
  
  * 然而，SI和EWC对于难到易的排序比容易到难的排序显示出意想不到的更好的结果。
  
  * 总结：任务顺序几乎对模型性能没有影响

### 6.2.6 定性比较

![图片9.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/04/23-10-06-49-%E5%9B%BE%E7%89%879.png)

* EBLL和PackNet的训练计算时间增加了一倍（见浅绿色），因为它们都需要额外的训练步骤，分别用于自动编码器和压缩阶段

* GEM需要反向传播获取样本集梯度。 iCaRL最近邻分类器在测试阶段耗时

* PackNet和HAT需要给定样本的任务标识符来加载适当的掩码和嵌入，因此不符合任务不可知

* 重放方法iCaRL 和GEM存储原始图像，有隐私问题

## Que

- 持续学习和增量学习的区别：增量学习是对持续学习进行限制

- 提出了一个**新的框架**来确定continual learner的稳定性和可塑性的权衡

- 在多个不同模型上，对比**模型容量**、**正则化方式**、**真实环境（数据集分布）**、**任务顺序**对增量学习的影响。
