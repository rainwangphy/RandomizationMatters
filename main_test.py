import torch.cuda
import torch.nn as nn
import numpy as np
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(3))
# print(torch.cuda.)
print(torch.cuda.is_available())