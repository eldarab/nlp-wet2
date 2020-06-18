from preprocessing import ParseDataReader
from model import DnnParser
import torch
from torch import nn

x = torch.rand(3, 2)
softmax = nn.Softmax(0)
print(softmax(x[0]))
