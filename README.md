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
        $$
        X^TX\theta - X^Ty =0 \\
         X^T\theta = X^Ty \\
         \theta = (X^TX)^{-1}X^Ty
        $$
        

#### Gradient Descent 梯度下降

> 梯度下降的核心思想是通过一个微调参数进行迭代，来获得一个cost function的最小值

+ 在这一过程中，我们首先给$\theta$ 一个随机值，然后每一次调整一步来实现更小的``cost function``比如MSE，知道这个算法收敛到了一个最小值

+ 如果``learning rate``设置的过于小，那么训练的速度就会很慢，反之，如果learning rate特别高，就有可能没法找到一个好的答案

+ 事实上，cost function的形状基本都是碗状的，但在他的特征们都有不同的scales时，就会是``enlongated bowl``(椭圆形的加长碗)

  + 在标准的碗状梯度下降算法中，算法可以更快的速度获取到最小值，然而在加长椭圆形的碗装形态中，它可能会首先进入一个局部最小值，然后需要花费更长的时间才能获得到全局最小值

  + 这就是为什么要使用标准化，归一化等

    

#### Batch Gradient Descent 批梯度下降

> 一次计算所有的数据，因此这个计算方法的速度相当慢

+ 要计算出一次需要下降多少，我们需要计算MSE的偏导数
  $$
  \frac{\partial}{\partial\theta_j}MSE(\theta)
  = \frac{2}{m}\sum^m_{i=1}(\theta^Tx^{(i)-y^(i)})x^{(i)}_j \\
  \nabla MSE(\theta) = \frac{2}{m}X^T(X\theta-y) \\
  \theta^{(next step)} = \theta - \eta\nabla MSE(\theta)
  $$
  

+ 如上公式中$\eta$ 是learning rate学习率
+ Python实现Batch Gradient Descent

```python
eta = 0.1 #learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) # random intialization

for iteration in range(n_interations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= eta * gradients
```

#### Stochastic Gradient Descent 随机梯度下降

+ 在使用随机梯度下降的时候，我们应该让训练实例保持独立并且是``identically distributed``来保证可以实现全局最优，实现这个的一个方法就是在训练的时候随机抽取一部分数据集，或者是在每一个epoch都重新打乱数据

#### Mini-batch Gradient Descent

+ 每次计算选取一部分数据集，相对于随机梯度下降来说，最大的优点是可以利用Mini-Batch可以获得来自硬件的优化，尤其是使用GPU的时候



### Polynomial Regression

+ Learning Curve
  + 通过观察learning curve中train和val两条线之间的距离可以获得是否过拟合



### Regularized Linear Models 正则化

#### Ridge Regression 岭回归

+ 岭回归是一种线性回归的在正则化版本，他在cost function中增加了一个``regularization term正则项``
  + 正则项的公式
  +  $\alpha\sum^n_{i=1}\theta^2_i$
    + 在加入正则项后，为了实现最小的cost function，算法需要同时兼顾拟合数据，也要考虑到正则项对成本函数的影响
    + 超参数 $\alpha$ 用于控制我们需要正则化模型的程度
    + 岭回归cost function: $J(\theta) = MSE(\theta)+\alpha \frac{1}{2}\sum^n_{i=1}\theta^2_i$
    + 需要注意的是，bias term $\theta_0$是Joni是不会被正则化的，总和从i=1开始
  + 这一正则项强制让学习算法不只是去拟合数据，同时也保持模型的权重尽可能得小
  + Ridge Regression closed-form solution
    +  $\hat\theta = (X^TX+\alpha A)^{-1}X^Ty$
      + 其中，A是除了左上角值为0以外的单位矩阵
  + 设置Penalty = $l2$ 则意为使用l2 范式的平方的一半，即为使用简单的岭回归

#### Lasso Regression

+ ``Least Absolute Shinkage and Selection Operator Regression``

+ Lasso回归是另一种线性回归的正则化，它添加了一个正则项，但使用$l1$ norm
  + Lasso Regression cost Function
    + $J(\theta)=MSE(\theta)+\alpha\sum^n_{i=1}|\theta_i|$
    + 和岭回归不同，lasso只是用了$\theta_i$的绝对值，而不是平方的一半

+ Lasso回归的一个关键特征就是，它会倾向于排除掉最不重要的特征的权重
  + 它会自动执行一个特征选择，并输出一个稀疏模型（只有少数非零权重参数）



#### Elastic Net

+ Elastic Net 是岭回归和Lasso回归的中间项
+ 他的正则项就是简单的岭回归加Lasso回归，再附上一个mix ratio $r$
  + $r=0$的时候就是岭回归，$r=1$ 的时候就是Lasso

#### 如何在不同的正则化项目之间做选择呢？

+ 通常来说，我们要尽可能避免朴素线性回归，正则化多少需要一点
+ ``岭回归``会是一个比较好的``默认选项``，但如果认为只有一部分特征是有用的，那么就选择使用Lasso或者Elastic Net
+ 通常来说，``Elastic Net是优先于Lasso``的，因为当特征数量大于训练集数量时或有一些特征之间的关系特别明显时，Lasso可能会表现得不太稳定



#### Early Stopping

+ 另一种进行正则化例如梯度下降模型的方式就是只要我们的``validation error``到达了一个最小值，就停止训练，这被称作``early stopping``
+ 设置```warm_start=True``即为每一次模型都在上一次训练的基础上进行训练
