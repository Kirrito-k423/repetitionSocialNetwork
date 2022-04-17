import torch

x = torch.tensor([1, 2, 3, 4, 5])
# print(x.t())
# x = x.view(1,-1).transpose_(0, 1)
x = x.view(1, -1).t()
print(x)

