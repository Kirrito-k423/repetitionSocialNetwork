import keyword
import torch

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()
meta = []
while len(meta) < 100:
    meta = meta + keyword.kwlist  # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = i

label_img = torch.rand(100, 3, 10, 50)
for i in range(100):
    label_img[i] = i / 100.0

writer.add_embedding(torch.randn(100, 2), metadata=meta, label_img=label_img)
# writer.add_embedding(torch.randn(100, 5), label_img=label_img)
# writer.add_embedding(torch.randn(100, 5), metadata=meta)
