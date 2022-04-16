import torch.nn as nn
import torch


input1 = torch.rand(10, 5)
input2 = torch.rand(10, 5)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

cos2 = nn.CosineSimilarity(dim=0, eps=1e-6)

cos10 = nn.CosineSimilarity(dim=-1, eps=1e-6)
cos11 = nn.CosineSimilarity(dim=-2, eps=1e-6)
output = cos(input1, input2)

print(output)
output = cos2(input1, input2)

print(output)

output = cos10(input1, input2)

print(output)

output = cos11(input1, input2)

print(output)
