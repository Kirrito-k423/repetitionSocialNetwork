import torch
import torch.nn as nn

# With Learnable Parameters
m = nn.BatchNorm1d(10)
# Without Learnable Parameters
m = nn.BatchNorm1d(10, affine=False)
input = torch.randn(20, 10)
output = m(input)
print(input)
print(output)
