import torch.nn as nn
import torch


input1 = torch.randn(10, 5)
input2 = torch.randn(10, 5)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)

print(output)
