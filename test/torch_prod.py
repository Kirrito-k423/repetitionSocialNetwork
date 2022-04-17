import torch

x = torch.randn((2, 2))
x = torch.tensor([[5, 6], [1 / 5, 2]])
print(x)
print(torch.prod(x, 0))  # product along 0th axis

