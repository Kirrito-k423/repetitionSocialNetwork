import torch

predict = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
predictDelOne = 1 - predict
lossPredict = torch.cat((predictDelOne, predict), 0).t()
print(lossPredict)

