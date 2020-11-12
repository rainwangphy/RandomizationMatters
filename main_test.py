import torch
import torch.nn as nn
import numpy as np

a = torch.zeros([2, 4, 5])
a = nn.Parameter(a)
print(a)

a = -np.inf

if 3 > a:
    a = 3
    print(a)
    print("True")
