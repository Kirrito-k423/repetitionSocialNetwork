# repetitionSocialNetwork

## To Do: 

1. 没有用COSINESIMILARITY计算 f
   1. https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
   2. 网络中自己定义的tensor，optimizer会自动更新他们的值吗？还是需要手动来
   3. 之前detach的问题是，对于一个tensor，在backward前修改了。导致问题
2. batchsize的使用原理
   1. 什么时候归一化，还是归一化是隐含
3. 测试要迭代H