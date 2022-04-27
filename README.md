# repetitionSocialNetwork

2022春季USTC社会计算实验二： 社交预测建模 + 针对数据进行预测建模 


## database

上传了最后筛选过后的两个数据集

```
nodeNum_10613.tmp # 各个城市中group数最大的数据集子集
nodeNum_1770.tmp # 在上一个子集的要求上，额外去除掉参与group数小于3的用户的数据集子集
```

### 数据集简单分析

```
python3 ./src/dataAnalysis.py 1770
```

## Run

### help

```
$ python3 ./src/main.py -h
usage: main.py [-h] [-m {all,predict,skipall}] [-c {cuda:0,cuda:1,cuda:2}] [-n NODE] [-l {Mse,CrossEnt,Focal}]
               [-d {debug,NotDebug}] [-b BATCH] [-lr LR] [-e EPOCH]

please enter some parameters

optional arguments:
  -h, --help            show this help message and exit
  -m {all,predict,skipall}, --mode {all,predict,skipall}
                        just predict
  -c {cuda:0,cuda:1,cuda:2}, --cuda {cuda:0,cuda:1,cuda:2}
                        which gpu to use
  -n NODE, --node NODE  train nodenum
  -l {Mse,CrossEnt,Focal}, --loss {Mse,CrossEnt,Focal}
                        loss kinds
  -d {debug,NotDebug}, --debug {debug,NotDebug}
                        use debug example data
  -b BATCH, --batch BATCH
                        batch size
  -lr LR, --learn LR    train learning rate
  -e EPOCH, --epoch EPOCH
                        epoch num
```

### 训练并测试

当前目录，BatchSize=32，需要大约20GB显存。

```
python3 ./src/main.py -m all -c cuda:0 -n 1770 -l Focal -d NotDebug -b 32 -lr 0.001 -e 200
```

### 只测试

```
python3 ./src/main.py -m predict -c cuda:0 -n 1770 -l Focal -d NotDebug -b 32 -lr 0.001 -e 200
```

## 已经训练好的网络参数

位于 saveNet 下

## 文档和相关链接

pdf 位于 resource文件夹

Notion临时文档：https://shaojiemike.notion.site/2-83e02db271714832be55f3a44d0cf326

## To Do:

1. 更光滑的激活函数效果可能更好
2. 对于阈值h的迭代公式，由于是乘积，假如其中某一项是负数，整个效果就颠倒了
3. 数据集还需要进一步均衡