OpenCV is a very traditional method on CV problems.

Tensorflow is a deep learning method.

### Image Fundamentals

gray image (0,255)

color image: (R,G,B) -->  three channels

- width, height, channel

#### Traditional image processing:

人工定义算子(Operator) (对应位置相乘相加：卷积运算)

CNN: 卷积神经网络，计算机自动学习算子

#### The basic building blocks of convolution

全连接神经网络 （DNN）

flatten -> dense -> dense -> softmax

用DNN处理图像有两个问题：

1. 参数量过大：

   如果我们使用一张mnist 28*28的图片, flatten之后是一个784D的向量，第二层用一个64D的 neural, 第三层32D的 neural, 第四层10D的 neural, 再做一个 softmax. 最终这张图片经过这样一个DNN会生成一个10维的向量，这个向量就是我们最终的预测值。

   784是这张图像像素的维度，如果我们这张图片的第二个全连接有n个神经元，那一共就有784 * n个参数。In this case, 28 * 28 is a small image, we might have some hd image like 1000 * 1000, only first layer would be too large.

2. 像度之间相关性大

   因为第二层与第一层有连接，使得像素之间的相关性就很大。DNN assumes all the pixels have some certain 相关性，which is 不合理的。

   所以我们引入了CNN.

### CNN 

- 卷积层 （下面两个feature 解决了之前DNN处理图像的两个问题）
  - 局部感受野 
  - 共享参数
- 池化层

#### 局部感受野：感受只是<u>局部的</u> 

不讨论所有像素点，只是讨论局部的pixel. 

#### Some important concepts:

- channel (有几个matrix 叠加，叠加的个数就是 channel)
- padding (在外面一圈添加0，使下一层神经元满足一定条件。填充多少需要用公式计算)
- strides (在用卷积做运算的时候，阴影滑动的格数)
- kernel size（小图的阴影大小，一般是3x3，（一般选择奇数，因为奇数拥有对称性）
- filters
- feature map
- Receptive field (感受野) (看我的感受野到底会受多少层的影响，蕴含了多少层的信息)
  - 两个参数：
    - SAME （不填充padding）
    - VALID (填充 padding)

#### 1*1 kernel:

- 压缩通道
- 扩展通道

#### 池化层

- 平均池化
- 最佳池化



#### CNN构架设计原则：

1. feature map 越来越小
2. channel 越来越大

### The classical convolution structure 

**Imagenet比赛**

CNN -> Lenet -> Alexnet ->

- Microsoft track: go deeper

  VGG -> Resnet -> Resnetv2

- google track: go wider

  InceptionV1 -> InceptionV2 -> InceptionV3

-> InceptionV4 (小而精)

### Lenet

lenet 作为 CNN 应用在真实场景的开端

![](https://raw.githubusercontent.com/ffflora/data-science-notes/master/archived-pics/open-course/lenet.png)

#### Alexnet

第一次应用在工业界真实的很大的数据集上，imagenet, 是千分类的数据集。

![](https://raw.githubusercontent.com/ffflora/data-science-notes/master/archived-pics/open-course/alexnet.png)

加入了 dropout 机制

#### Inception 结构

![](https://raw.githubusercontent.com/ffflora/data-science-notes/master/archived-pics/open-course/inception.png)

基本思想：引入了【感受野】，把不同感受野的信息拼凑在一起。

需要做一些padding，因为不同感受野的 kernel size 不同，但我们需要 make sure the sizes are matched.

大的kernel size 都可以被很多小的kernel size 叠加代替。

#### VGGNet

go deeper 

#### Resnet 

![](https://raw.githubusercontent.com/ffflora/data-science-notes/master/archived-pics/open-course/resnet.png)

为何有效？

- 能解决梯度消失的问题
- highway （类似一个集成模型）
- ODE 的数值近似

#### Xception （keras)

![](https://raw.githubusercontent.com/ffflora/data-science-notes/master/archived-pics/open-course/xception.png)

