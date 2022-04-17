# repetitionSocialNetwork

## To Do: 

1. 上GPU
2. GNN PyG 的 mini-batching

## 其他
3. 没有用COSINESIMILARITY计算 f
   1. https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
   2. 网络中自己定义的tensor，optimizer会自动更新他们的值吗？还是需要手动来
      1. 加入是输入的tensor，backward确实会传递梯度，但是optimizer不会修改值
      2. optimizer = optim.Adam([var1, var2], lr = 0.0001)自己添加自己的参数
      3. optimizer = optim.SGD([ {'params': model.base.parameters()}, {'params': model.classifier.parameters(), 'lr': 1e-3} ], lr=1e-2, momentum=0.9)
         1. 以上optim.SGD()中的列表就是构建每个参数的学习率，若没有设置，则默认使用最外如：model.base.parameters()参数使用lr=1e-2  momentum=0.9
      4. 或者自己添加added to the model as a Parameter.
         1. self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
   3. 之前detach的问题是，对于一个tensor，在backward前修改了。导致问题
4. batchsize的使用原理
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
5. 测试要迭代H
6. **message(应该需要x)**
7. **GPU数据传输**