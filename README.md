# MachineLearning-Notebook
机器学习读书笔记 参考书目《机器学习实战》、《Python深度学习》

- [MachineLearning-Notebook](#machinelearning-notebook)
- [Chapter 1 机器学习概览](#chapter-1-机器学习概览)
  - [机器学习类别](#机器学习类别)
    - [监督学习 Supervised Learning](#监督学习-supervised-learning)
    - [无监督学习 Unsupervised learning](#无监督学习-unsupervised-learning)
    - [半监督学习 Semisupervised Learning](#半监督学习-semisupervised-learning)
    - [强化学习 Reinforcement Learning](#强化学习-reinforcement-learning)
    - [Batch and Online Learning](#batch-and-online-learning)
    - [Instance-Based Versus Model-Based Learning](#instance-based-versus-model-based-learning)
  - [Chapter 2 Classification 分类](#chapter-2-classification-分类)
    - [ROC Curve](#roc-curve)
    - [MultiClass Classification 多分类问题](#multiclass-classification-多分类问题)
    - [Multilabel Classification 多标签分类问题](#multilabel-classification-多标签分类问题)
    - [Multioutput Classification](#multioutput-classification)
  - [Chapter 3 Training Models](#chapter-3-training-models)
    - [Linear Regression](#linear-regression)
# Chapter 1 机器学习概览

## 机器学习类别

### 监督学习 Supervised Learning
+ K-Nearest Neighbors K近邻模型
+ Linear Regression
+ Logistic Regression
+ Support Vector Machines (SVMs) 支持向量机
+ Decision Trees and Random Forests
+ Neural networks

### 无监督学习 Unsupervised learning
+ ``聚类 Clustering``
  + K-Means
  + DBSCAN
  + Hierachical Cluster Analysis层次聚类分析
+ ``异常检测和新颖性检测`` Anomaly detection and novelty detection
+ > 异常检测模型会用于检测是否有``奇怪的数据``出现，比如在信用卡交易中防止诈骗行为，此外，也可以用来在把数据集放进训练模型前移除掉``离群值Outliers``
+ > 新颖性检测就旨在检测出和训练集中相比明显是新的实例的情况
  + One-class SVM
  + Isolation Forest

+ ``可视化与降维 Visualization and dimensionality reduction``
+ > 在可视化算法中，给他们一系列复杂的，没有标注的数据，他们会将输入的数据集，转换成容易``通过2D或者3D图表展现``出来的形式，这样可以帮助我们数据是如何被组织的，同时也可以帮助我们分辨没有被检测到的样式
  + Principal Component Analysis (PCA) 主成分分析法
  + Kernel PCA
  + Locally Linear Embedding (LLE)
  + t-SNE t-Distributed Stochastic Neighbor Embedding
+ ``关联规则学习`` Association rule learning
+ > 关联性学习的目标是为了挖掘数据不同属性之间的关系
  + Apriori
  + Eclat

### 半监督学习 Semisupervised Learning
> 数据集中只有一部分是有标签的，大部分的半监督学习都是由无监督学习和有监督学习算法组合而成的，比如
+ Deep Belief Networks(DBNs) 是基于无监督学习Restricted Boltzmann Machines(RBMs)的

### 强化学习 Reinforcement Learning
> 训练一个Agent，Agent将对给定的环境做出行为，我们则对他做出的行为正确与否给出奖励和惩罚进行训练

### Batch and Online Learning
+ ``Batch Learning``
  + > Batch learning中，它必须使用所有可用的数据进行学习，然后这会花费比较多的时间的资源，所以一般会在线下进行运算，通常也被称作``Offline Learning``
  + 如果我们希望一个batch learning system 能够了解一些新的数据（比如一种新的诈骗邮件形式），就需要重新训练整个数据集并重新把模型上线
+ ``Online Learning``
  + > 线上学习中，我们逐步地将数据实例放入系统中，分散成``mini-batches``，每一个学习步骤都会比较快并且节省资源，所以系统能够在线上就学习到新的数据

### Instance-Based Versus Model-Based Learning
> 还有一种给机器学习系统分类的方法就是查看他们是如何生成的
+ ``Instance-based learning``
  + 系统通过比较新数据和已学习数据的相似度来生成结果
+ ``Model-Based Learning``
  + 通过数据集的实例来生成一个模型，然后通过模型来进行预测

## Chapter 2 Classification 分类

+ Preicision
  + Precision = TP/(TP+FP)
  + 精确度就是模型器判断为正的数据集中，有多少是正确的
+ recall
  + recall = TP(TP+FN)
  + 召回率就是在所有应该被判定为正的数据集中，有多少被找了出来
+ F1 score
  + 通常，我们将精确率和召回率结合，F1 score通常是他们的调和平均数，调和平均数会对更小的值给予更大的权重，所以，只有在模型recal和precision都比较高的时候，才会获得一个不错的F1 score
$$
F_1 = \frac{2}{\frac{1}{precision}+\frac{1}{recall}}=2\times{\frac{precision\times recall}{precision + recall}}
$$

+ 通常来说，我们没有办法提升精确率的同时增加召回率，这被称作``precision/recall trade-off``
  + 在随机梯度下降模型SGDClassifier中，它会根据自己的``decision function``给每一个实例计算一个分数，如果那个分数超过了自己的``threshold临界点``，那么就把它标记为positive类。
  + Sklearn库没有办法让我们直接设置threhold，但是我们可以使用decision_function()方法来使用我们想使用的threhold临界值
  + 此外，我们可以使用``precision_recall_curve()``方法来计算对于所有可能的threhold临界值所对应的精确度和召回率

### ROC Curve
+ ``Receiver operating characteristic(ROC)``，ROC曲线是用于二元分类器的常用工具，它和precision/recall curve非常相似，它会展示出``True positive rate``和``false positive rate``，前者就是recall rate，后者则是阴实例被错误判定成正确实例的概率
  + 这里同样存在取舍，召回率越高，就会产生更高的false positive rate，我检测正样本的能力变强了，但也同时把更多的负样本归类进来了
+ 一种衡量分类器的方法就是比较``area under the curve(AUC)``
+ 选择ROC curve还是PR(precision_recall_curve)呢？

### MultiClass Classification 多分类问题
+ 我们有不同的策略来基于多个二元分类模型实现多分类模型
  + 比如我们现在需要训练一个系统来区分0-9这十个数字
  + 1. 训练十个二元分类模型，给每一个数字各训练一个，当我们需要进行分类的时候，每一个二元分类器都会给数字一个decision score，然后我们选取最高分数的作为这个数字的结果，这通常被称作``one-versus-the-rest (OvR)/(One-versus-all)``策略
  + 2. 第二种策略是给数字的所有组合都设计一个二元分类器，比如0和1之间，0和2之间，1和2之间，以此类推。
    + 这通常被称作``one-versus-one(OvO)``策略，如果我们有N个类，那么我们则需要训练次数如下
      + $$ N\times(N-1)/2 $$

### Multilabel Classification 多标签分类问题
+ 有些情境下，我们需要分类器给一个实例输出多个类别，每个类别都是二元的。
+ ```y_multilabel = np.c_[y_train_large, y_train_odd]```

### Multioutput Classification
+ 在此情景下，每个类别都可能有多种可能的值
+ 比如给一个图像做去噪处理，图像上每一个元素点的输出值都可以在0-255之间浮动，所以是多输出分类

## Chapter 3 Training Models

### Linear Regression
$$
\hat{y}=\theta_0+\theta_1 x_1 +\theta_2x_2 +\cdot\cdot\cdot+\theta_n x_n \quad ①\\
\hat{y}=h_\theta(x)=\theta\cdot x=\theta^Tx \quad ②
$$

+ 在如上公式中，theta是参数向量，包含了偏移量theta0 与特征值的比重theta1-thetaN
  + x则是实例的特征向量，包含了x0到xN，其中x0一直等于1，以此实现纵轴的偏移
  + 向量间相乘使用的是点乘法，②中点乘展开后即得到了①

+ 线性回归是如何进行训练的

  + 我们使用``Root Mean Square Error (RMSE)``平均根方差来衡量模型对训练集的拟合度，因此，在训练一个线性回归模型的过程中，我们只需要找到一个 $\theta$ 可以最小化RMSE的即可

  + 在实际的应用中，我们让模型契合更小的MSE均方差会更简单

  + ``MSE cost function for a Linear Regression model``

    + $$
      J(\theta) = MSE(X,h_\theta)=\frac{1}{m} \sum^{m}_{i=1}(\theta^Tx^{(i)} - y^{(i)})^2
      $$

    + 为了计算能最小化MSE的 $\theta$ ，这里存在一个``closed-form solution``，也就可以可以通过一个数学等式直接计算出结果，通常被称作``Normal Equation正交方程`` 
    
      + $$ \hat{\theta} = (X^TX)^{-1}X^Ty $$
      + $\hat{\theta}$ 是能够最小化cost function的值，y是所有y值的向量
    
    + 正交方程的推导过程
    
      + 特征数据$X$被组织成维度为$m\times(n+1)$ 的设计矩阵，第一列全部都是1，对应截距，其中$m$是样本的数量，$n$ 是特征的数量，$y$ 是 $(m\times 1)$ 的标签数据
      + 要求成本函数的最小值，则要令$ \frac{\partial}{\partial\theta}J(\theta) = 0$
      +  

