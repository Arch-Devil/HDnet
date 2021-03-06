# HDnet


《人工智能》助教课材料

数据集链接：https://pan.baidu.com/s/1BvYKDQS1nhGA30hFrvBo-Q 
提取码：sbaw 

实验文档链接：https://pan.baidu.com/s/1aY7TYDdLKbydp807-B_Ziw 
提取码：0chk

预训练模型链接：https://pan.baidu.com/s/1jFHmm4dzxLNvZ8xjwcLadQ 
提取码：sz51 

PyTorch下载链接：https://pan.baidu.com/s/16RnFuGrDp8Nf3bsvRgQ3Fg
提取码：pxnh

CUDA下载链接：https://pan.baidu.com/s/1oYqi2rRinrYWcXwFIjymrA
提取码：9uyr

Anaconda下载链接：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2019.07-Windows-x86_64.exe

PyCharm下载链接：https://download.jetbrains.8686c.com/python/pycharm-professional-2018.3.5.exe

## 文本生成训练步骤
0、下载代码,解压,用pycharm打开项目

1、输入命令，爬取数据文本，也可以直接下载文本数据，放在/data/comment目录下
```
python crawl.py
```

2、打开文件"conf/lstm.yaml"，调整参数

3、运行程序，训练模型
```
python train.py --config_path "conf/lstm.yaml"
```


## 人脸分类训练步骤
0、下载代码,解压，将数据集放到/data目录下,用pycharm打开项目

1、Pycharm打开菜单View->Tool Windows->Terminal，依次输入以下命令
```
pip install opencv_python==4.4.0.42 -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install albumentations==0.4.6 -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install geffnet==0.9.8 -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

2、输入命令，分割数据集
```
python tools/data_preprocess.py
```

3、打开文件"conf/resnet50.yaml"，调整参数

4、如果pretrained为True，需要下载预训练模型，链接在下方，文件放到/models目录下

5、运行程序，训练模型
```
python train.py --config_path "conf/resnet50.yaml"
```

6、训练结束，测试模型
```
python test.py --config_path "conf/resnet50.yaml"
```


