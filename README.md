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

# Chapter 2 Classification 分类

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

# Chapter 3 Training Models

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
+ 设置``warm_start=True``即为每一次模型都在上一次训练的基础上进行训练





### Logistic Regression

> 通常来说逻辑回归普遍用于预估一个实例属于某一个类别的概率，如果预测概率大于50%则归为那一类

#### Estimating Probabilities

+ 一个逻辑回归模型会首先计算所有输入特征的权重和（加上bias term），并输出一个概率值
  + 预估概率公式 $\hat p = h_\theta(x)=\sigma(x^T\theta)$
  + 如上中的$\sigma$是一个S型的``sigmoid function`` 输出一个0-1之间的数字
    + 公式如下 $\sigma(t)=\frac{1}{1+exp(-t)}=\frac{1}{1+e^{-t}}$

#### Training and Cost Function

+ 训练的目标是为了获取参数向量 $\theta$ 让模型给结果为1的向量更高的概率，给0的向量更低的概率

  + 这一训练目的由以下逻辑回归的``Cost Function``实现(针对一个单独的训练实例)
    $$
    c(\theta)=\left\{
    \begin{aligned}
    -log(\hat p)\quad if\quad y=1 \\
    -log(1 - \hat p)\quad if\quad y=0
    \end{aligned}
    \right.
    $$

  + 针对整个训练集的cost function可以是所有训练集实例的平均cost，可以被写成一个单独的表达式，也被称作``log loss``
    $$
    J(\theta)=-\frac{1}{m}
    \sum^m_{i=1}
    \big[y^i log(\hat p^i)+(1-y^i)log(1-\hat p^i)
    \big]
    $$

  + 然而，并没有一个closed-form equation闭式方程可以直接计算出能够让cost function最小的 $\theta$ 值。

  + 但这个cost function是convex状的，所以可以使用梯度下降或其他最优方程来找到最小值，同样的，我们只需要寻求cost function的 $j^{th}$ 模型的参数 $\theta_j$ 偏导数即可，公式如下
    $$
    \frac{\partial}{\partial\theta_j}
    J(\theta) = \frac{1}{m}
    \sum^m_i (\sigma(\theta^T x^i)-y^i)x^i_j
    $$

#### Decision Boundaries

> 正负样本的判断概率都是50%的点位就是决策边界
>
> 逻辑回归也可以添加正则项，Scikit-Learn默认添加了$l_2$正则项，强度控制不使用$\alpha$ 而是使用 $C$ ，和线性回归相反，C越大，正则化的力度越小



#### Softmax Regression

+ 逻辑回归模型可以直接用于生成有多个类的模型(multiple classes)，不需要将几个二元分类器结合在一起

+ 这被称作``Softmax Regression, or Multinomial Logistic Regression``

  + 实现方法：当给定一个实例x的时候，Softmax Regression model首先会给每一个类$k$ 计算分数$s_k(x)$ ，然后使用``softmax function(or called normalized exponential)`` 计算每一个类的概率

  + 计算$s_k(x)$ 的公式和线性回归预测的公式非常相似
    $$
    s_k(X)=X^T\theta^k
    $$

  + 每一个类class都有一个自己的专用参数向量 $\theta^k$ ，所有的这些向量都储存在``parameter matrix``的行中

  + 在我们计算完实例x属于每一个类的分数后，就可以通过使用softmax function预估概率

    + 这个函数会计算每一个分数的exp，然后正则化他们（除以所有exp的总和）

    + 分数通常被称作``logits`` or ``log-odds`` 

    + Softmax function
      $$
      \hat p_k=\sigma(s(x))_k
      =\frac{exp(s_k(x))}{\sum_{j=1}^Kexp(s_j(x))}
      $$

      + 如上等式中
        + K是类的数量
        + s(x) 是一个包含了示例x对于每一个类的分数

  + 和逻辑回归分类器相似，Softmax Regression分来其同样也是将预估概率最高的类作为预测结果

    + Softmax Regression classifier prediction
    
  + 训练过程中，为了最小化Cost Function，我们需要降低交叉熵``cross entropy``, 当模型给我们的目标类预估了一个低概率的时候，交叉熵就会惩罚模型
  
    + 交叉熵经常被用于衡量预估类的概率集合和目标类的匹配度
  
    + Cross entropy cost function
      $$
      J(\Theta)=-\frac{1}{m}
      \sum_{i=1}^{m}
      \sum^K_{k=1}
      y^{(i)}_k
      \log(\hat{p}_k^{(i)})
      $$





# Chapter 4. Support Vector Machines 支持向量机

+ 支持向量机是一个强大，并且多功能的模型，能够用于线性、非线性分类，回归或是异常检测。
  + SVM尤其适用于复杂中小量样本的分类问题



## Linear SVM Classification

+ SVM分类器会拟合在类之间最宽的可能的street（街道），这被称作``large margin classification`` 
  + 要注意的是，在这个street外的任何训练实例都不会影响到决策边界，street只会被它的边缘实例所影响，而这些实例就被叫做``support vectors``支持向量
  + SVM对数据集中特征的尺度(scale)相当敏感，应该在使用模型前做标准化的预处理

### Soft Margin Classfication

+ 如果让我们严格要求所有的实例都必须要在street外的右侧，这就被称作``hard margin classification``硬边界分类。
  + 硬边界分类存在两个主要的问题
    + 1. 它只对线性可分的数据有效果
      2. 它对离群值(outliers)特别敏感
+ 为了解决硬分类边界的问题，我们使用soft margin classification
  + 我们的目标是在``让street尽可能大的情况下限制住margin violations边缘错误实例``，margin violation指的是那些出现在street内的实例，或是出现在错误方向的实例
  + 在使用Sickit-Learn的时候，我们可以设定一个超参数C，在SVM过拟合的时候，可以降低C来正则化



## Nonlinear SVM Classification

+ 当我们遇到一个只有一个特征x1且线性不可分的数据集的时候，我们可以给其添加一个特征x2=(x1)^2，这就可以让他变得线性可分了

+ 以上转化就是PolynomialFeatures，在Sklearn库中的transformer内，可以设置degree

  + 在一个pipeline内，我们添加如下模型，就可以实现对线性不可分的数据集的拟合

    ```python
    Polynomial Features(degree=3),StandardScaler(),LinearSVC(C=10, loss='hinge')



### Polynomial Kernel

+ 上面提到的方法没有办法处理一个非常复杂的数据集，并且会有一个相当高的维度和非常多的特征数量，这会让这个模型的速度变得很慢

+ 但是，我们在使用SVM的时候，可以尝试着去使用一个非常神奇的数学技巧叫做``kernel trick``核方法

  + 核方法可以让我们不用添加高维的多项式特征就可以得到相同的结果，这可以避免特征数量爆炸式增长

    ```python
    poly_kernel_svm_clf = PipeLine([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(X,y)
    ```

  + 如上代码中，我们让SVM分类器使用了一个degree为3的多项式核，degree可以用于控制拟合程度，coef0用于控制，（相比于低维多项式），模型被高维多项式影响的程度



### Similarity Features

+ 另一个用于解决线性不可分问题的方法，是使用``similarity function``相似函数来增加参数

  + 相似函数衡量每个实例与特定``landmark地标``之间的相似程度。

  + 比如，我们有一个一维数据集，有两个landmark地标 $x_1=-2, x_1=1$ 

    ​	首先，我们定义这个相似函数为``Gaussian Radial Basis Function(RBF)``高斯径向基功能，同时$\gamma = 0.3$ 

    + Gaussian RBF 钟形分布曲线
      $$
      \phi_\gamma(x,l) = \exp(-r||x-l||^2)
      $$
      $l$ 是landmark地标，那我们如何选择地标呢？最简单的方法是给每一个实例都创建一个地标，通过这个方法可以创建很多维度，这样就可以增大让数据集变成线性可分的概率



### Gaussian RBF Kernel

+ RBF kernel同样可以用在SVM内，但是计算花费的时间特别多，尤其是那些数量级很大的数据，但是使用Kernel核方法就可以避免这一情况出现

  ```python
  rbf_kernel_svm_clf = Pipeline([
      ('sclaer', StandarSclaer()),
      ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
  ])
  rbf_kernel_svm_clf.fit(X,y)
  ```

  

+ 其他也有一些kernel会被运用到，有一些kernel是为一些特别的数据结构特制的，比如``String kernels``通常用于分类text documents or DNA sequences
+ 既然又怎么多类别的kernel，那具体如何进行选择呢
  + linear kernel 是我们的首选，而且LinearSVC() 要比 SVC(kernel='linear')更快，尤其复杂数据集
  + 如果数据集不是很大，也可以尝试使用RBF kernel以及其他的特殊kernel



### 

## SVM Regression

+ 想要让SVM来实现回归的任务，我们只需要让目标反过来
  + 让尽可能多的点分布在street内，同时限制边缘的异常值margin violations
  + street 的宽度通过超参数 $\epsilon$ 进行控制



## 原理

### Decision Function and Predictions

+ SVM分类模型要判断一个实例就是通过decision function计算对应的结果，然后和0,1进行比较

  + Decision Function
    $$
    \hat{y}=
    \left\{
    \begin{aligned}
    	0\ \ if\ w^Tx+b<0\\
    	1\ \ if\ w^Tx+b\ge0
    \end{aligned}
    \right.
    $$

## Training Objective 训练目标





# Decision Trees 决策树



## Making Prediction

+ 实际上决策树不需要进行太多的数据预处理，他们不需要特征标准化或中心化

+ 决策树中的每一个节点的``gini``基尼系数都会衡量它的``impurity``不纯度

  + 当一个节点的基尼系数等于0的时候，它是``pure``纯的

  + Gini Impurity
    $$
    G_i = 1 - \sum_{k=1}^nP_{i,k}^2
    $$
     其中，$P_{i,k}$ 是第i个节点所有训练实例中类别k实例的概率

    ScikitLearn使用的是 CART算法，CART算法只创建二元树，非子节点总会有两个子树（问题的答案永远是Yes/No）其他的算法比如ID3 会创建的子树大于2

+ White BOX verses Black BOX

  + 决策树是相当直观的，他们的决策都比较容易进行解释，这样的模型被我们称作``white box model``白盒模型
  + 相对应的，随机森林或是神经网络通常被认为是黑盒模型，他们通常有很好的表现，但是很难对他们的预测进行解释



## The CART Training Algorithm

CART: ``Classification and Regression Trees`` 是Sickit-Learn使用的生成树的算法

+ CART 算法工作的过程中会首先会使用单独的一个特征 $k$ 和一个极限值threshold $t_k$ 来把整个训练集拆分成两个子集。

  + 那么，这个特征以及极限值是如果选择的呢？

    + 它会首先搜索所有的配对 $(k,t_k)$ , 在所有的配对中，找到最纯的purest子集(根据他们的尺寸分配权重)

    + 算法尝试最小化的Cost Function如下
      $$
      J(k,t_k)=
      \frac{m_{left}}{m}G_{left}+
      \frac{m_{right}}{m}G_{right}\\
      $$
       其中 $G_{left/right}$ 衡量左/右子集的不纯度，$m_{left/right}$ 是左右子集中的实例的数量，

      成本函数中实际上就是针对左右子集不纯度计算了一个加权和

+ 当CART算法将模型成功分为两个子集之后，它会用同样的逻辑给子集继续做分割操作
  + 直到它触及了最深的分裂限度或没有办法通过分割子集来降低不纯度了，就停止。
+ CART算法是一个``greedy algorithm`` 贪婪算法，它不一定能够实现最优的结果 ，然而如果我们需要找到最优的结果，可能会需要$O(\exp{(m)})$ 的运算时间，这让问题变得相当难以处理



## Gini Impurity or Entropy

基尼不纯度还是熵

+ 默认情况下，模型会使用基尼不纯度，但我们也可以通过更改超参数``critierion = 'entropy'``来使用熵

  + 当一个数据集只包含一个类别的实例的时候，他的熵就是0

  + 第i个节点的熵
    $$
    H_i = -\sum^{n}_{\ \ k=1\\P_{i,k}\ne0}P_{i,k}\log_2(P_i,k)
    $$

+ 所以我们选用基尼不纯度还是信息熵呢？
  + 大部分情况下，使用两者不会产生明显的区别，基尼不纯度会有更快的计算速度，会是一个比较好的默认选择。
  + 此外，基尼不纯度倾向于将最频繁的类和分支隔离开，然而熵会倾向于创建一个更平衡的树



## Regularization Hyperparameters

+ 像决策树这样，不对训练数据集有太多猜想的模型，通常被称作``nonparametric model``非参数模型，不是因为它没有任何参数，而是因为在训练前没有确定的参数数量，所以模型可以紧密地贴合数据。
+ 相对应的，``parametic model``参数模型，比如线性模型，有一个确定好的参数数量，所以他们的``degree of freedom``自由度是固定的，这会降低他们过拟合的风险。
+ 决策树这样的模型是非常容易过拟合的，我们需要在训练过程中限制决策树的自由度，也就是进行正则化
  + 限制树模型的最大深度``max_depth``，``min_sample_split``, ``min_sample_leaf``, ``min_leaf_nodes``
  + 增加min_*参数，降低max_*参数就可以实现正则化了
+ 其他的算法首先不加限制得去训练决策树，然后再进行剪枝``pruning``，减掉那些没有必要的节点。
  + 如果一个节点得所有子类都是叶子节点，且他的纯度提升是不够统计显著的，我们就认为它是不必要的节点，应该被剪掉。统计显著测试可能会使用到 $X^2-test$ 卡方检测。 



## Regression

+ 决策树同样可以用于进行回归任务，不过相比于预测一个类别，回归中它会预测一个值

+ 回归任务中使用CART算法得使用逻辑和分类是类似的，只是现在它尝试着在区分数据集的时候，最小化MSE均方差，cost function如下所示，为左子树和右子树的均方差的加权和

  + CART cost function for regression
    $$
    J(k,t_k) = \frac{m_{left}}{m}MSE_{left}+
    \frac{m_{right}}{m}MSE_{right}
    $$



## Instability 不稳定性

决策树的一些局限性

+ 1. 决策树喜欢执行正交决策边界，这让他们对训练集的旋转(rotation)比较敏感，有时候会带来无意义的消耗，通常来说一个好的解决办法时使用主成分分析``Principal Component Analysis`` ，这会让我们获得一个有更好的方向``orientation``的训练集
  2. 另一个问题是，决策树对于训练集中的一些小的变量比较敏感，随机森林可以通过加权投票限制这个不稳定情况





# Ensemble Learning and Random Forests



## Voting Classifiers

+ 假设我们现在有了一批分类器模型，把这些模型的预测聚合在一起，并且将被投票最多的那个类作为结果，这样的分类器就会被称作``hard voting classifier``硬投票分类器
+ 当不同的预测模型之间是尽可能互相独立的时候，集成学习的效果会更好。比如在学习的时候对不同的模型使用完全不同的学习模型，这会提高整体预测精度。
+ 如果所有分类器都可以预估类别的概率，那么我们称它为``soft voting``，软投票模型通常可以获得更好的表现，因为他给更确定的票一个更高的权重（比如0.99相比0.55更重要）



## Bagging and Pasting

+ 另外一种获取一批预测模型的方法是，使用相同的算法，但是使用训练集中不同的子集进行训练
  + 当这个采样是涵盖``replacement重置的时候（有放回采样），被称作``bagging`` ``, short for ``bootstrap aggregating``
  + 相反，无放回采样的训练方式被称作``pasting`` 粘贴



## Extremely Randomized Trees

+ 如果我们想要让树变得更加随机，我们可以使用随机的极限值thresholds（再加上使用随机特征）
+ 这种极度随机的树，我们称它为``Extremely Randomized Trees (Extra-Trees)``
  + 这种技术实际上使用更多的偏差来换取方差``more bias for a lower variance``



## Feature Importance

+ 随机森林的另一个重要品质是他们可以很容易衡量每个特征的相对重要性
  + Scikit-Learn通过观察每个子节点通过那个特征降低的不纯度的加权平均值
  + 每个特征的重要性相加的总和为1



## Boosting

+ 最开始称为``hypothesis boosting`` 
+ Boosting 的想法就是逐步的``sequentially``训练模型，每一次训练都尝试修正之前的模型



### AdaBoost

+ 其中一种修正之前的模型的方法就是对那些欠拟合的数据实例提供更多的关注，这会让新的训练器越来越关注那些比较难的实例，这就是AdaBoost使用的技巧

+  这种序列化的学习技术，是不能平行进行的，因为每一个预测器都基于之前的训练结果 so it does not scale as well as bagging or pasting

+ AdaBoost的算法

  + 首先，每个实例的权重 $w^{(i)}$ 都初始化设置为 $\frac{1}{m}$

  + 在第一个预测器被训练之后，获得他的权重错误率 $r_1$

    + $Weighted\ error\ rate\ of\ the\ j^{th}\ predictor$
      $$
      r_j=\frac
      {\sum^{m}_{i=1\\\hat y_j^{(i)}\ne y^{(i)}}w^{(i)}}
      {\sum_{i=1}^{m}w^{(i)}}
      \\\\
      where\ \hat y_j^{(i)}\ is\ the\ j^{th}\ predictor's\ prediction\ the\ i^{th}\ instance
      $$

  + 预测器的权重 $\alpha_j$ 通过如下公式计算

    + $Predictor\ weight$
      $$
      \alpha_j=\mu\log\frac{1-r_j}{r_j}
      $$

  + 其中 $\mu$ 是学习率learning rate, 预测器的准确度越高，它的权重就越大，如果只是随机的猜测，那么它的权重就会使接近0的，如果整体倾向于是错的，那他就是负数

  + 然后，AdaBoost模型就会通过下面的等式来更新实例的权重

    + $Weight\ update\ rule$
      $$
      \rm{for}\ i=1,2,...,m\\
      w^{i}\gets
      \left\{
      \begin{aligned}
      	w^{(i)}\  if\ \hat{y_j}^i=y^{(i)},\\
      	w^{(i)}exp(\alpha_j)\ if\ \hat{y_j}^i\ne y^{(i)}
      \end{aligned}
      \right.
      $$



### Gradient Boosting

+ 与AdaBoost类似，GBRT也是逐步增加预测器，并且在先前的基础上进行训练。但是，它不调整实例的权重，而是尝试着让新预测器去拟合之前的预测器的``residual errors`` 残差

+ 一个简单的实现

  ```python
  from sklearn.tree import DecisionTreeRegressor
  
  tree_reg1 = DecisionTreeRegressor(max_depth=2)
  tree_reg1.fit(X,y)
  
  #现在我们训练第二个决策树，但将训练数据集设置为第一个预测器的残差residual errors
  y2 = y - tree_reg1.predict(X)
  tree_reg2 = DecisionTreeRegressor(max_depth=2)
  tree_reg2.fit(X,y2)
  
  #现在我们训练第三棵树，训练数据集设置为第二个预测器的残差
  y3 = y2 - tree_reg2.predict(X)
  tree_reg3 = DecisionTreeRegressor(max_depth=2)
  tree.reg3.fit(X,y3)
  
  #现在我们有个三棵树，我们可以将它们简单的聚合在一起
  y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg2))
  ```



## Stacking

+ ``Stacked generalization`` 相比于使用琐碎的函数来聚合预测结果，我们也可以选择训练一个模型来执行这个聚合

+ 我们将三个训练好的模型聚合为最终的一个模型，我们称它为``blender``搅拌机或是``meta learner``
+ 一个通用的训练融合器的方法是使用一个``hold-out set``保留集





# Chapter 7 Dimensionality Reduction 降维

+ 1. 提高训练速度
  2. 对于数据可视化非常有帮助



## 降维的主要途径

1. projection 投影
2. Manifold Learning 流形学习



### 投影

+ 在现实生活中的问题，训练集通常不是均匀地分布在所有维度的，有些特征一直是常数，而另一些特征是相关连的，所以实际上所有的数据集实例都在一个更低维度的子空间



### 流形学习

瑞士卷是一个典型的2D ``manifold流形``

+ 一个2D流形是一个在高维空间被弯曲和扭曲的形状。
+ 很多降维算法都通过训练数据集所在的流形进行建模，这就被称作``Manifold Learning``。这依托于``manifold assumption``流形假设，也被称作``manifold hypothesis``



## PCA

``Principal Component Analysis``会首先识别最接近数据集的超平面，然后向平面上投影数据。



+ Preserving the variance & Principal Components 

+ 为找到训练集的主成分``principal components``，使用标准矩阵分解方法：``Singular Value Decomposition(SVD)``，使用此方法可以将训练集矩阵X拆解为  $U \sum V^{T}$​​​ 三个矩阵相乘

  + 其中 V 包含了定义了我们寻找的主成分的单元向量

  + 使用Numpy的 ``svd()``函数来获取训练集的所有主成分，然后提出前两个主成分的单元向量

    ```python
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered)
    
    c1 = Vt.T[:, 0]
    c2 = Vt.T[:, 1]
    ```

  + 需要注意的是，PCA算法默认数据集是以原点为中心的， Sklearn会考虑数据集的中心，但自己实现PCA的过程中需要预处理数据分布情况。



#### Projecting Down to d Dimensions

+ 在我们确定好主成分后，就可以通过投影的方式进行降维至d维

+ 使用原矩阵X 乘以 ``V矩阵的前n列``

  ​	$\rm{X}_{d-proj} = \rm{X}\rm{W}_{d}$​

  ```python
  W2 = Vt.T[:, :2] #选取V矩阵的前d列
  X2D = X_centered.dot(W2) #原矩阵点乘V矩阵的前d列
  # X2D便是降维后我们获得的数据集
  ```

+ 我们可以把降维后的数据重构回原来的维度，但这会与原来的数据集必然存在区别，这被称作``reconstruciton error``重构误差

+ Sklearn算法中默认使用PCA算法，计算复杂度远小于使用``SVD()``算法



### Incremental PCA (IPCA)

+ 原有的PCA算法的缺点在于需要一个完整的数据集来让算法运行。

  + ``incremental PCA自增PCA``允许将训练集拆分成小的包，然后分别给每一个包使用一次IPCA，这对于规模非常大的数据集或是线上的PCA使用有很大帮助（比如在线上有时候会有新的实例出现）

    ```python
    from sklearn.decomposition import IncrementalPCA
    
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X_train, n_batches): #np.array_split()函数拆分训练集
        inc_pca.partial_fit(X_batch) # for循环逐一喂数据
    X_reduced = inc_pca.transorm(X_train)
    ```

    





### Kernel PCA

+ 在PCA中使用核方法可以在降维中 施展 复杂的非线性投射
  + 这有助于在投影后保留数据的聚类效果，或有时候可以使用在接近``twisted manifold扭曲流形``的数据集上



### LLE （Locally Linear Embedding）

另一种强大的``nonlinear dimensionality reduction(NLDR)`` 非线性降维方法。

+ 这是一种``Manifold Learning`` 流形学习技巧，不需要依赖任何的投影
+ LLE会首先计算训练集中每一个实例和起最近邻居的关联，然后寻找一个能够最好保留``local relationships``局部关系的低维表示
  + 这种方法使LLE针对``unrolling twisted manifolds展开瑞士卷流形`` 的效果尤其好



#### How LLE works:

1. 首先，针对每一个训练集实例 $\rm{x}^{(i)}$​，算法会先判断他的 $k$ 个最近邻，然后尝试去将 $\rm{x}^{(i)}$ 和他们的近邻视为线性函数去重构

2. 更确切的来说，它首先会找到一个权重 $w_{i,j}$​，使得 $\rm{x}^{(i)}$ 和 $\sum^{m}_{j}w_{i,j}\rm{X}^{(j)}$​ 的平方距离``squared distance`` 尽可能最小。所以LLE的第一步是一个``constrained optimization problem`` 约束优化问题，W就是一个权重矩阵，涵盖了所有的小权重$w_{i,j}$​.
   $$
   \rm{W}=argmin_{(W)}\sum^{m}_{i=1}(x^{(i)}-\sum^{m}_{j=1}w_{i,j}x^{(j)})^{2}\\
   \rm subject\ to\ \left\{w_{i,j} = 0\ if\ x^{(j)}\ is\ not\ oen\ of\ the\ k\ c.n. of;
   \ \sum^m_{j=1}w_{i,j}=1\ for\ i=1,2,···,m\right\}
   $$



3. 在获得权重$\rm{W}$后，我们就可以把训练集匹配进入d维空间中，同时尽可能地保留这些数据之间的局部关系；如果$z^{i}$是$x^{(i)}$在这个d维空间中的图像，那么我们就希望 $z^{(i)}$ 和 $\sum^{m}_{j=1}w_{i,j}\rm{Z}^{j}$​ 之间的平方距离(squared distance)尽可能最小。
   + 这与前一步非常相似，但前一步是在让实例保持固定，寻找最优化的权重，现在是尝试着做最大的保留：权重固定，但是找到每个实例在低维空间的最优位置





### Other Dimensionality Reduction Techniques

+ Random Projections
  + 使用``random linear projection`` 随机先行投影来把数据投影到更低维度。听起来比较疯狂，但实际上随即投影的效果对于保留距离非常好。降维的质量取决于实例的数量和目标维度。``sklearn.random_projection``
+ Multidimensional Scaling(MDS) 
  + 在降维的同时视图保留实例之间的距离
+ Isomap
  + 创建一个连接每个实例和他的近邻之间的图，然后在降维的同时尝试去保留实例之间的 ``geodesic distances`` 

+ t-Distributed Stochastic Neighbor Embedding (t-SNE)
  + 在降维的同时试图让相似的实例保持接近，让不同的实例保持分开，这主要用于可视化，尤其是高维数据集的聚类
+ Linear Dscriminant Analysis(LDA线性判别分析)
  + 是一个分类随附按，但是在训练的过程中，他学习到了类之间最有效的判别轴 ``discriminative axes`` ，并且这些轴(axes) 可以用于定义一个用于投射数据的超平面hyperplane
  + 使用这个方法的好处是：它的投影可以让类之间的距离尽可能的远，所以LDA是一个在使用其他分类算法（比如SVM）之前降维的好技巧。![image-20210821200923762](C:\Users\zleoliu\AppData\Roaming\Typora\typora-user-images\image-20210821200923762.png)







# Chapter 9. Unsupervised Learning Techniques



## Clustering

### K-Means

+ 在Kmeans的计算过程中，如果随机的初始化步骤效果不好，可能会导致收敛出局部最优解，对应的解决方法如下

+ Centroid initialization methods

  + 通过提高``centroid initialization质心初始化``的效果来降低风险

    ```python
    goo_init = np.array([[-3,-3],[-3,-2].[-3.-1],[-1,2],[0,1]])
    kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)
    ```

+ 另一个解决办法是使用不同的随机参数运行算法多次然后保持一个最优的结果。

  + 可以使用超参数n_init来控制，默认等于10；模型会首先运行十次，然后呈现最优解

+ 衡量KMeans聚类的效果：使用``model`s inertia``: mean squared distance between each instance and its closet centroid. ``所有实例距离他最近的中心的平均平方距离``



+ 一个KMeans的显著进步算法：**K-Means++**, proposed in a 2006 paper by David Arthur and Sergei Vassilvitskii.
  + 他们介绍了一种更聪明的初始化步骤：试图在选择初始中心点的时候让他们互相保持远离``tends to select centroids that are distant from one another``
  + 这让K-Means算法更不容易收敛出一个次优解 ``much less likely to converge to a suboptimal solution``
  + K-Means++的初始化算法流程如下
    1. 选取一个中心点centroid $c^{(1)}$​​，在数据集中随机均匀的选择``chosen uniformly at random``
    2. 选取一个新的中心点centroid $c^{(i)}$, 随后使用一个分布概率（距离已有中心点的距离越远，概率越高），选择出后续的中心点
    3. 重复之前的步骤知道所有的k个中心点都被选取完成
  + KMeans会默认使用这种方法



### Accelerated K-Means and mini-batch K-Means

+ 2003年Charles Elken的papaer中对K-Means算法带来巨大的改进
+ 通过避免一些没必要的距离计算来提升算法的运算效率
  + 利用三角不等式来实现``triangle inequality`` （两点之间直线最短），同时持续跟踪实例和中心点之间距离的长、短边界.





### Finding the optimal number of cluster寻找到最优的聚类数目

+ 计算各个数目k对应的inertia，寻找其中的拐点``elbow``

+ inertia的能力有限，如果真的要计算，还得是看轮廓系数``sihouette coefficient``的平均值：``silhouette socre``轮廓分数
+ 一个实例的轮廓系数 = $\rm(b-a)/{max}(a,b)$​ ，其中—— a 是到类内每一个实例的平均距离，b是到最近邻聚类的平均距离。
  + 轮廓系数的值域从-1 到 1 不等；
  + 轮廓系数接近 +1 说明这个实例在自己的蔟类的中央，并且远离其他的蔟类
  + 轮廓系数接近 0 说明这个实例处于一个蔟的边界
  + 轮廓系数接近 -1 说明这个实例有可能处在一个错误的边界
+ 一个更加能表示出信息的可视化方法是展示出每一个实例轮廓系数，根据他们的蔟类和各自的系数值进行排序。这被称作``silhouette diagram``





### Limits of K-Means 

+ 当Kmeans面对变化很大的尺寸、不同密度、或者是非球面形状时，表现可能会欠佳

![image-20210822155616028](C:\Users\zleoliu\AppData\Roaming\Typora\typora-user-images\image-20210822155616028.png)

面对如上的椭圆蔟类``elliptical clusters``时，高斯混合模型``gaussian mixture models``可能会有更好的效果





### DBSCAN

Another popular clustering algorithm that illustrates a very different approach based on ``local density estimation``. 

+ 这种算法将聚类定义为连续的高密度区域，它的工作原理如下：
  + 首先，针对每一个实例，算法会先计算有多少实例在他们的一段很小的距离 $\epsilon$​ 区域内。这个区域被称作实例的 $\epsilon-neighborhood$ 
  + 如果一个实例至少有参数 ``min_samples``个实例 ，那他就会被认为是``core instance``，也就是，核心实例是分布于那些高密度区域的``dense regions`` 
  + 所有核心实例的近邻都会被认为是属于同一个cluster。这些近邻们可能会包含其他的核心实例，因此，一长串的近邻核心实例形成了单个蔟
  + 任何不是核心实例、且他的近邻中没有核心实例 的实例，会被认为是一个``anormaly`` 异常
+ 这种算法在此种情况下效果比较好：所有的聚类都有足够的密度，并且他们被低密度区域很好地划分开

![image-20210822162307447](C:\Users\zleoliu\AppData\Roaming\Typora\typora-user-images\image-20210822162307447.png)

+ 针对DBSCAN调整合适的epsilon来适应不同的数据集情况





### 其他的聚类算法

+ Agglomerative clustering
  + 聚类的层次是由下往上逐步搭建的；在每一个迭代的过程中，agglomerative clustering将最近的一对蔟类连接在一起。它能够创建一个灵活的并且能够呈现相当多信息的聚类树``flexible and informative cluster tree`` ，并且能够捕捉不同形状的聚类。
+ BIRCH (Balanced Iterative Reducing and Clustering using Hierachies)
  + BIRCH算法是针对数据规模特别大的数据集的，并且在获得相似结果的前提下，可以比batch K-Means更快，只要特征列的数量不要太多（小于二十）。这个算法允许他使用比较有限的内存来处理规模很大的数据集。





## Gaussian Mixtures

> Gaussian mixture model(GMM) 是一个概率模型``probabilistic model`` ，它假设实例都是从几个不确定参数的高斯分布的混合中生成出来的。

所有从同一个cluster中的单个高斯分布生成的模型都会像是一个椭圆体``ellipsoid`` 。当我们观察一个实例的时候，没有办法知道他来自哪一个高斯分布，也不知道这个分布的具体参数都是什么。



+ ``Expectation-Maximization(EM)``最大期望算法，和KMeans算法有很多相似的地方：
  + 首先会随机初始化聚类的参数，然后会重复两个步骤直到收敛
    + 1. 首先将实例分配给蔟类（这被称为 **expectation step**
      2. 然后更新蔟类（这被称为 **maximization step**











# Deep  Learning



























































