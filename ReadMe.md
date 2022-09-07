### 环境

- python3.8 （一开始是3.6，但是后来到差分隐私发现opacus必须用py3.7以上，于是直接换了环境）
- pytorch 1.10.2
- CPU：AMD Ryzen 5 4600H with Radeon Graphics            3.00 GHz
- 编辑器使用pycharm

### 文件结构

![image-20220831001324397](ReadMe/image-20220831001324397.png)

plot：存储图像
graph.py：从四个txt读取数据，绘制图像

### 实验记录

参数；non-IID, E = 1, B = 10, model = mnist_cnn, learning_rate = 0.01, num_comm = 50, num_of_clients = 600, C = 0.1

#### 未添加差分隐私

<figure class="half">
	<div>
    	<img src="./plot/acc0.png", title="Accuracy", width = "50%", align = left>
    </div>
	<div>
    	<img src="./plot/loss0.png", title="Loss", width = "50%", align = right>
	</div>
</figure>

#### 添加噪声

参数：sensitivity = 1, max_norm = 4

##### eps = 0.5

100%|██████████| 60/60 [00:14<00:00,  4.02it/s]

<figure class="half">
	<div>
    	<img src="./plot/acc(eps=0.5).png", title="Accuracy", width = "50%", align = left>
    </div>
	<div>
    	<img src="./plot/loss(eps=0.5).png", title="Loss", width = "50%", align = right>
	</div>
</figure>

##### eps = 2

100%|██████████| 60/60 [00:14<00:00,  4.21it/s]

<figure class="half">
	<div>
    	<img src="./plot/acc(eps=2).png", title="Accuracy", width = "50%", align = left>
    </div>
	<div>
    	<img src="./plot/loss(eps=2).png", title="Loss", width = "50%", align = right>
	</div>
</figure>

##### eps = 8

100%|██████████| 60/60 [00:13<00:00,  4.36it/s]

<figure class="half">
	<div>
    	<img src="./plot/acc(eps=8).png", title="Accuracy", width = "50%", align = left>
    </div>
	<div>
    	<img src="./plot/loss(eps=8).png", title="Loss", width = "50%", align = right>
	</div>
</figure>

#### 汇总

<figure class="half">
	<div>
    	<img src="./plot/trainAccuracy.png", title="train", width = "50%", align = left>
    </div>
	<div>
    	<img src="./plot/testAccuracy.png", title="test", width = "50%", align = right>
	</div>
</figure>

<figure class="half">
    <div>
        <img src="./plot/trainLoss.png", title="train",  width = "50%", align = left>
    </div>
    <div>
    	<img src="./plot/testLoss.png", title="test", width = "50%", align = right>
    </div>
</figure>


#### 总结

一般而言，epsilon越小，隐私保护越好，但是加入的噪声就越大，从上面最后四组图像也可以看出epsilon越大的准确度越高，同时损失下降的越快，下界越低；训练时间方面，从每秒迭代次数可知epsilon越大训练越快。


### 参考内容

https://github.com/zergtant/pytorch-handbook
https://blog.csdn.net/qq_36018871/article/details/121361027
https://blog.csdn.net/xbn20000224/article/details/120660384
https://zhuanlan.zhihu.com/p/359060612
https://github.com/pytorch/opacus
https://blog.csdn.net/qq_36965067/article/details/123604987
https://stackoverflow.com/questions/66994662/pytorch-warning-about-using-a-non-full-backward-hook-when-the-forward-contains-m
https://blog.csdn.net/qq_38242289/article/details/80798952?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-80798952-blog-77477833.topnsimilarv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-80798952-blog-77477833.topnsimilarv1&utm_relevant_index=12
https://blog.csdn.net/niunai112/article/details/113739841?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-113739841-blog-80798952.pc_relevant_multi_platform_whitelistv6&spm=1001.2101.3001.4242.2&utm_relevant_index=4





