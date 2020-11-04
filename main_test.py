import torch
import torch.nn as nn

a = torch.zeros([2, 4, 5])
a = nn.Parameter(a)
print(a)
