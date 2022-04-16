import torch

x = torch.tensor([1, 2, 3])
print(x.expand(10, 3))
print(x.expand(4, -1))

