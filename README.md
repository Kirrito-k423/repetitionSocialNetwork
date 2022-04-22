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



## 其他
3. 没有用COSINESIMILARITY计算 f
   1. https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
4. 网络中自己定义的tensor，optimizer会自动更新他们的值吗？还是需要手动来
      1. 加入是输入的tensor，backward确实会传递梯度，但是optimizer不会修改值
      2. optimizer = optim.Adam([var1, var2], lr = 0.0001)自己添加自己的参数
      3. optimizer = optim.SGD([ {'params': model.base.parameters()}, {'params': model.classifier.parameters(), 'lr': 1e-3} ], lr=1e-2, momentum=0.9)
         1. 以上optim.SGD()中的列表就是构建每个参数的学习率，若没有设置，则默认使用最外如：model.base.parameters()参数使用lr=1e-2  momentum=0.9
      4. 或者自己添加added to the model as a Parameter.
         1. self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True)).cuda()
   1. 之前detach的问题是，对于一个tensor，在backward前修改了。导致问题
5. batchsize的使用原理
   1. 什么时候归一化，还是归一化是隐含。
      1. https://stackoverflow.com/questions/51735001/how-to-include-batch-size-in-pytorch-basic-example
      2. 网络后会自动归一化吗？
      3. 不会。在计算层和非线性层间加入BN层。可以更好的发现特征
   2. BN层只是效果会变好
   3. 由于矩阵操作，增加batch/行号。每行经过同一个网络，引起的就是输出行号增加。只需要对每行单独计算出来的误差进行sum或者mean得到一个误差值，就可以反向传播，训练参数。
   4. 简单来说就是平均了一个batch数据的影响，不会出现离谱的波动，方向比较准确。
   5. **但是对于GNN。batch in GNN pyG可行吗？**
      1. 为此引入了ADVANCED MINI-BATCHING来实现对大量数据的并行。
      2. https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
      3. 图像和语言处理领域的传统基本思路：通过 rescaling or padding(填充) 将相同大小的网络复制，来实现新添加维度。而新添加维度的大小就是batch_size。
      4. 但是由于图神经网络的特殊性：边和节点的表示。传统的方法要么不可行，要么会有数据的重复表示产生的大量内存消耗。
      5. 在PyG里引入了一种新方法来实现
      6. 实现：
         1. 邻接矩阵以对角线的方式堆叠(创建包含多个孤立子图的巨大图)
         2. 节点和目标特征只是在节点维度中串联???
6. 测试要迭代H
7. **message(应该需要x)**
8. **GPU数据传输**
   1. net to device。其中的参数也会上cuda
      1. tensor([[0.7201],
        [0.4810],
        [0.6461],
        [0.0099]], device='cuda:0', requires_grad=True)
   1. x.to(device)不是移动，而是复制。所以一定要接受y=x.to(device)
   2. 基于GPU上数据产生的数据也在GPU上
9.  耗时
   3. 总耗时 cPython https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script
   4. 正常情况应该是前向和后向时间占大头
   5. https://pytorch.org/docs/stable/profiler.html
10. 缩短data加载耗时
    1.  有个dali好像是做这个的，不过我没有用过
    2.  多弄几个worker
        1.  dataloader可以设置num worker
        2.  加载数据多线程
        3.  不过这些大概都不解决to这一步的耗时
    3.  以及用pin memory

11. 其他
   6. https://blog.csdn.net/qq_44015059/article/details/114747640