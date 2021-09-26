# Simple_OCR

# 环境配置：

python == 3.6

tensorflow == 2.0.0

h5py == 2.10.0

opencv-python == 4.5.3.56


# 代码介绍：

img文件夹：用来存储验证码图片，运行data_simulate.py文件即可生成10000张带标记的验证码数字图片。

Logs文件夹：用来存储模型训练权重。

第1步：运行data_simulate.py文件，生成训练数据集，并存储至img文件夹。

第2步：运行main.py文件，训练模型，并对验证集进行检测。
    
第3步：visualize.py文件，网络可视化，显示不同分支分类模型的关键识别像素区域，生成heatmap_1-6.jpg。

这些热力图代表着，不同分支的CNN对原始图像关注的区域。


# 算法效果：

训练10个epoch后val loss达到瓶颈，val loss = 1.98左右在也无法降低。

整个序列识别精度只有65%，但单个字符识别精度93%。
