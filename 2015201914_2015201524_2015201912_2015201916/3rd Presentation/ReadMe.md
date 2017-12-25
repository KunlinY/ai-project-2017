# AI Car 第二次展示报告
## 实现的功能
- 基于摄像头的双目测距
- 基于卷积神经网络的图像识别
- 基于长短期记忆网络的音频识别
- 实时学习
- 听从人命令
- 跟随主人速度行走
- 为主人导航
## 功能介绍
###室内模式
- 启动小车，人在小车侧面晃动，小车学习主人身体特征——实时学习
- 小车识别出主人之后，带领主人在室内随机游走
- 时刻紧跟主人脚步，主人停下就停下，主人放慢脚步就减慢速度——实时学习
- 距离障碍物半米时向右拐弯——双目测距
###跑道模式
- 主人喊出口令“Hey AI”，小车识别出来之后学习主人身体特征——音频识别
- 认识主人后，启动小车马达——实时学习
- 识别跑道白线，顺着跑道线外延行走——图像识别
- 同时跟着主人速度走——实时学习
## 详细算法介绍
### 双目测距
双目测距原理图如下：
![](http://img.my.csdn.net/uploads/201010/24/0_1287878649g0L5.gif)

双目测距主要是利用了目标点在左右两幅视图上成像的横向坐标直接存在的差异，也就是视差d，由相似三角形原理得知其与目标点到成像平面的距离Z存在着反比例的关系 Z = fT/d。

**相机标定**： 摄像头由于光学透镜的特性使得成像存在着径向畸变，可由三个参数k1,k2,k3确定；由于装配方面的误差，传感器与光学镜头之间并非完全平行，因此成像存在切向畸变，可由两个参数p1,p2确定。单个摄像头的定标主要是计算出摄像头的内参（焦距f和成像原点cx,cy、五个畸变参数（一般只需要计算出k1,k2,p1,p2，对于鱼眼镜头等径向畸变特别大的才需要计算k3））以及外参（标定物的世界坐标）。而双目摄像头定标不仅要得出每个摄像头的内部参数，还需要通过标定来测量两个摄像头之间的相对位置（即右摄像头相对于左摄像头的旋转矩阵R、平移向量t）。
![](https://latex.codecogs.com/gif.latex?x_p = x_d\left(1+k_1r^2+k_2r^4+k_3r^6\right))
![](https://latex.codecogs.com/gif.latex?y_p = y_d\left(1+k_1r^2+k_2r^4+k_3r^6\right))
![](https://latex.codecogs.com/gif.latex?x_p = x_d+\left[2p_1y+p_2\left(r^2+2x^2\right)\right])
![](https://latex.codecogs.com/gif.latex?y_p = y_d+\left[p_1\left(r^2+2y^2\right)+2p_2x\right])

**双目校正**: 双目校正是根据摄像头定标后获得的单目内参数据（焦距、成像原点、畸变系数）和双目相对位置关系（旋转矩阵和平移向量），分别对左右视图进行消除畸变和行对准，使得左右视图的成像原点坐标一致、两摄像头光轴平行、左右成像平面共面、对极线行对齐。这样一幅图像上任意一点与其在另一幅图像上的对应点就必然具有相同的行号，只需在该行进行一维搜索即可匹配到对应点。

**双目匹配**: 双目匹配的作用是把同一场景在左右视图上对应的像点匹配起来，这样做的目的是为了得到视差图。双目匹配被普遍认为是立体视觉中最困难也是最关键的问题。得到视差数据，通过上述原理中的公式就可以很容易的计算出深度信息。

**实际注意事项**：
- SIFT特征提取算法对左右图像点提取特征
- knnMatch取k=2找到左右图片最佳匹配
- 再过滤去除坏的匹配点
- 对于剩下的点使用相似三角形计算公式得到图片各点景深标在图上
- 最终小车避障可根据其中少数点进行判断，或者取均值。

标定结果如下：
以下图片均为右转90度的结果，因为小车拍摄到的视频原状是右偏的
![墙角的实拍图](http://upload-images.jianshu.io/upload_images/3204092-5233e9a26624c50f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![拐角的实拍图](http://upload-images.jianshu.io/upload_images/3204092-be4f3589cc41e9fc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![信箱的实拍图](http://upload-images.jianshu.io/upload_images/3204092-9d12ff4cb2405ec2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![信箱的实拍图](http://upload-images.jianshu.io/upload_images/3204092-e90d805f68327b6d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由标定结果可以看出，其测距效果还是比较接近真实值的。
### 图像识别

#### CNN算法介绍
![](http://upload-images.jianshu.io/upload_images/8920871-4df70ba1699211d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

使用CNN神经网络对六百多张图片进行学习，判断小车应当直走、左转、还是右转。如左图所示，白线斜率过大，小车距离白线过近，因此小车应该左转，如中间图片所示，小车应该直走，如右图所示，视野内并没有白线，此时默认小车直走。
#### CNN神经网络的基本结构

![](http://upload-images.jianshu.io/upload_images/8920871-8a1060796aa069b4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出最左边的图像是输入层，计算机理解为输入若干个矩阵，接着是卷积层（Convolution Layer），在卷积层后面是池化层(Pooling layer)，卷积层+池化层的组合可以在隐藏层出现很多次，在若干卷积层+池化层后面是全连接层（Fully Connected Layer, 简称FC），最后是输出层。
1. 卷积层
卷积层是CNN神经网络中最重要的一层，我们通过如下的一个例子来理解它的原理。图中的输入是一个二维的3x4的矩阵，而卷积核是一个2x2的矩阵。这里我们假设卷积是一次移动一个像素来卷积的，那么首先我们对输入的左上角2x2局部和卷积核卷积，即各个位置的元素相乘再相加，得到的输出矩阵S的S00S00的元素，值为aw+bx+ey+fzaw+bx+ey+fz。接着我们将输入的局部向右平移一个像素，现在是(b,c,f,g)四个元素构成的矩阵和卷积核来卷积，这样我们得到了输出矩阵S的S01S01的元素，同样的方法，我们可以得到输出矩阵S的S02，S10，S11，S12S02，S10，S11，S12的元素。

![](http://upload-images.jianshu.io/upload_images/8920871-f4df5aaacfbf2428.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2. 池化层
池化层的作用是对输入张量的各个子矩阵进行压缩。假如是2x2的池化，那么就将子矩阵的每2x2个元素变成一个元素，如果是3x3的池化，那么就将子矩阵的每3x3个元素变成一个元素，这样输入矩阵的维度就变小了。

要想将输入子矩阵的每nxn个元素变成一个元素，那么需要一个池化标准。常见的池化标准有2个，MAX或者是Average。即取对应区域的最大值或者平均值作为池化后的元素值。

下面这个例子采用取最大值的池化方法。同时采用的是2x2的池化。步幅为2。

![](http://upload-images.jianshu.io/upload_images/8920871-723fe51b10311831.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

首先对红色2x2区域进行池化，由于此2x2区域的最大值为6.那么对应的池化输出位置的值为6，由于步幅为2，此时移动到绿色的位置去进行池化，输出的最大值为8.同样的方法，可以得到黄色区域和蓝色区域的输出值。最终，我们的输入4x4的矩阵在池化后变成了2x2的矩阵。进行了压缩。
3. 损失层
dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。

![](http://upload-images.jianshu.io/upload_images/8920871-84e202734475fffe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

dropout最重要的功能就是防止数据出现过拟合。
#### 算法具体实现
1. CNN结构图
使用keras搭建卷积神经网络

![](http://upload-images.jianshu.io/upload_images/8920871-7510f47e3e607f0f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2. CNN各层介绍
- 卷积层*2：3*3小核计算，降低复杂度同时不损失精度
- 激活层：Relu，f(x)=max(0,x)，收敛速度快
- 池化层：区域压缩为1/4，降低复杂度并减少特征损失
- 全连接层*2：将分布式特征表示映射到样本标记空间
- Dropout层：Dropout设为0.5，防止过拟合，减少神经元之间相互依赖
- 激活层：softmax，平衡多分类问题
3. 效果分析

![](http://upload-images.jianshu.io/upload_images/8920871-39dc901b83d08761.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图是我们各个类别的准确率和召回率。可以看出，除了类别1，也就是左转类的召回率较低以外，其他类的准确率和召回率都较高。

![](http://upload-images.jianshu.io/upload_images/8920871-21e2d32374808fcc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

宏平均（Macro-averaging），是先对每一个类统计指标值，然后在对所有类求算术平均值。
微平均（Micro-averaging），是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标。
4. 具体代码实现
```
model = Sequential()  

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),  
                        padding='same',  
                        input_shape=(200,480,1))) # 卷积层
model.add(Activation('relu')) #激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2  
model.add(Activation('relu')) #激活层  
model.add(MaxPooling2D(pool_size=pool_size)) #池化层
model.add(Dropout(0.25)) #神经元随机失活
model.add(Flatten()) #拉成一维数据
model.add(Dense(128)) #全连接层1
model.add(Activation('relu')) #激活层  
model.add(Dropout(0.5)) #经过交叉验证
model.add(Dense(nb_classes)) #全连接层2  
model.add(Activation('softmax')) #评分函数
  
#编译模型  
model.compile(loss='categorical_crossentropy',  
              optimizer='adadelta',  
              metrics=['accuracy'])  
#训练模型  
model.fit(train, y, batch_size=32, epochs=3,  
          verbose=1)
```

#### 实验结果
下图为本实验的ROC曲线，由此可见，除了左转之外，剩下的曲线AUC值均达到了0.97，最差的AUC值也达到了0.84，效果还算不错。
![ROC曲线](http://upload-images.jianshu.io/upload_images/3204092-54a2ada912dafb69.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
实验的分类报告如下：
Class | Precision | Recall | F1-Score
---------|------------|---------|---------
0 | 0.73 | 1.00 | 0.85
1 | 1.00 | 0.18 | 0.31
2 | 0.89 | 0.94 | 0.91
Avg/Total | 0.81 | 0.80 | 0.75

### 音频识别
音频识别的目标是识别出主人的口令“Hey AI”，识别出来口令中是否包含“Hey AI”，为一个二分类问题，使用的工具为RNN的一个变种LSTM。
#### LSTM介绍
Long Short Term 网络—— 一般就叫做 LSTM ——是一种 RNN 特殊的类型，可以学习长期依赖信息。LSTM 由[Hochreiter & Schmidhuber (1997)](https://link.jianshu.com?t=http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)提出，并在近期被[Alex Graves](https://link.jianshu.com?t=https://scholar.google.com/citations?user=DaFHynwAAAAJ&hl=en)进行了改良和推广。在很多问题，LSTM 都取得相当巨大的成功，并得到了广泛的使用。

LSTM 通过刻意的设计来避免长期依赖问题。记住长期的信息在实践中是 LSTM 的默认行为，而非需要付出很大代价才能获得的能力！

所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。
![标准 RNN 中的重复模块包含单一的层](https://upload-images.jianshu.io/upload_images/42741-9ac355076444b66f.png)
LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于 单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。
![LSTM 中的重复模块包含四个交互的层](https://upload-images.jianshu.io/upload_images/42741-b9a16a53d58ca2b9.png)

LSTM 的关键就是细胞状态，水平线在图上方贯穿运行。

细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。

![](https://upload-images.jianshu.io/upload_images/42741-ac1eb618f37a9dea.png)

LSTM 有通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。

Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”！

LSTM 拥有三个门，来保护和控制细胞状态。

#### LSTM具体实现
LSTM的框架图如下
![LSTM框架图](http://upload-images.jianshu.io/upload_images/3204092-58fc28ff1d697061.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

首先使用了两层双向LSTM，接着利用Flatten实现到两层全连接层的过渡。虽然单向LSTM已经足够进行分类，但为了获得更高的准确度，是用了更强的双向LSTM。

最后实验结果在测试集上的各项指标均接近于1，也就是全部分类正确，就不进行图表绘制。

### 实时学习
#### 算法介绍
1. 识别出运动的像素点

![](http://upload-images.jianshu.io/upload_images/8920871-99f9717936f2bd72.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过对比相邻的两帧图像之间像素点的移动，标注出移动的像素点。得到效果图如下图所示。

![](http://upload-images.jianshu.io/upload_images/8920871-4ec9499c539d0303.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

代码如下所示：
```
def draw_flow(old, new, step=4):
    flow = cv.calcOpticalFlowFarneback(
        cv.cvtColor(old, cv.COLOR_BGR2GRAY), 
        cv.cvtColor(new, cv.COLOR_BGR2GRAY), 
        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = new.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
    fx, fy = flow[np.int32(y), np.int32(x)].T

    lines = np.int32(np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2))

    for (x1, y1), (x2, y2) in lines:
        if sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) < 15:
            continue
        cv.line(old, (x1, y1), (x2, y2), (0, 128, 0), 2)
        cv.circle(old, (x1, y1), 3, (0, 255, 0), -1)
        # x1 y1是old的运动点坐标，x2y2是new运动点的坐标
    return old
```
2. 画出目标区域
kmeans 算法接受参数 k ；然后将事先输入的n个数据对象划分为 k个聚类以便使得所获得的聚类满足：同一聚类中的对象相似度较高；而不同聚类中的对象相似度较小。聚类相似度是利用各中对象的均值所获得一个“中心对象”（引力中心）来进行计算的。

Kmeans算法是最为经典的基于划分的聚类方法，是十大经典数据挖掘算法之一。Kmeans算法的基本思想是：以空间中k个点为中心进行聚类，对最靠近他们的对象归类。通过迭代的方法，逐次更新各聚类中心的值，直至得到最好的聚类结果。

![](http://upload-images.jianshu.io/upload_images/8920871-8012bcd409797bcb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们使用Kmeans聚类分析的算法，将运动的像素点划分为三个类别，分别用矩形框将区域框出。

![](http://upload-images.jianshu.io/upload_images/8920871-86c64b93e10fb08b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3. 特征提取
我们使用颜色作为人的主要特征，找出上步标注出的三个矩形框中面积最大的一个，进行主颜色的提取。
```
def get_dominant_color(image):  
      
#颜色模式转换，以便输出rgb颜色值  
    image = image.convert('RGBA')  
      
#生成缩略图，减少计算量，减小cpu压力  
    image.thumbnail((200, 200))  
      
    max_score = 0
    dominant_color = 0
      
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):  
        # 跳过纯黑色  
        if a == 0:  
            continue  
          
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]  
         
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)  
         
        y = (y - 16.0) / (235 - 16)  
          
        # 忽略高亮色  
        if y > 0.9:  
            continue  
          
        score = (saturation + 0.1) * count  
          
        if score > max_score:  
            max_score = score  
            dominant_color = (r, g, b)  
      
    return dominant_color
```
4. 实时识别
我们根据颜色特征来识别出实时图像中人的位置。在RGB颜色空间中，以主颜色+-20作为判断的颜色区域，找出符合的像素点。通过erode和dilate来平滑像素点，得到一个区域，然后通过opencv的轮廓寻找功能找到区域轮廓的像素点，用矩形框标出这个区域。

![](http://upload-images.jianshu.io/upload_images/8920871-ec752bde2d3741dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
mask = cv2.inRange(image, lower, upper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

if len(cnts) > 0:  
#找到面积最大的轮廓  
	c = max(cnts, key = cv2.contourArea)
	x1,y1 = 1000,1000
	x2,y2 = 0,0
    
	for i in range(0,len(c)):
		if c[i][0][0] < x1:
			x1 = c[i][0][0]
		if c[i][0][0] > x2:
			x2 = c[i][0][0]
		if c[i][0][1] < y1:
			y1 = c[i][0][1]
		if c[i][0][1] > y2:
			y2 = c[i][0][1]

cv2.rectangle(image,(x1,y1),(x2,y2),(55,255,155),5)
```
## 小组分工
组员 | 参与工作
---- | ----
杨昆霖 | 图像识别、音频识别
刘珏 | 图像识别、实时学习
施畅 | 双目测距、数据采集
侯尚文 | 数据标注、视频拍摄
