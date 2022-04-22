# repetitionSocialNetwork

## database

随便上传了两个数据集
```
nodeNum_10613.tmp # 各个城市中group数最大的数据集子集
nodeNum_1770.tmp # 在上一个子集的要求上，额外去除掉参与group数小于3的用户的数据集子集
```
### 数据集简单分析
```
python3 ./src/dataAnalysis.py 1770
```

## Run
### 训练
当前目录
```
python3 ./src/main.py -m all -c cuda:0 -n 1770 -l Focal -d NotDebug -b 32 -lr 0.001 -e 200
```
### 只测试
```
python3 ./src/main.py -m predict -c cuda:0 -n 1770 -l Focal -d NotDebug -b 32 -lr 0.001 -e 200
```
##
## To Do: 


