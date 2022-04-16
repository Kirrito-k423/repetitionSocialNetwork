# repetitionSocialNetwork

## To Do: 

1. 没有用COSINESIMILARITY计算 f
   1. https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
   2. 网络中自己定义的tensor，optimizer会自动更新他们的值吗？还是需要手动来
      1. 加入是输入的tensor，backward确实会传递梯度，但是optimizer不会修改值
   3. 之前detach的问题是，对于一个tensor，在backward前修改了。导致问题
2. batchsize的使用原理
   1. 什么时候归一化，还是归一化是隐含。
      1. https://stackoverflow.com/questions/51735001/how-to-include-batch-size-in-pytorch-basic-example
      2. 网络后会自动归一化吗？
      3. 不会。在计算层和非线性层间加入BN层。可以更好的发现特征
   2. BN层只是效果会变好
   3. 由于矩阵操作，增加batch/行号。每行经过同一个网络，引起的就是输出行号增加。只需要对每行单独计算出来的误差进行sum或者mean得到一个误差值，就可以反向传播，训练参数。
   4. 简单来说就是平均了一个batch数据的影响，不会出现离谱的波动，方向比较准确。
   5. **但是对于GNN。batch in GNN pyG可行吗？**
3. 测试要迭代H
4. **message(应该需要x)**