import torch

A = torch.tensor([[1, 2, 3, 4, 5, 6]])

splitTensorList = torch.split(A, 3, 1)
print(splitTensorList)
# tmp=torch.zeros(1,)
# for i in splitTensorList:

