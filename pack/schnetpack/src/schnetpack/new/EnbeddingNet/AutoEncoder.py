import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk
from schnetpack import Properties

class AutoEncoder(torch.nn.Module):
    def __init__(self, in_Size=128, out_Size=64, inter_Size=96,
                 em_len=64, radical=48, angular=4):
        super(AutoEncoder, self).__init__()
        self.in_size = in_Size
        self.out_size = out_Size
        self.inter_size = inter_Size
        self.em = nn.Embedding(100, em_len, padding_idx=0)
        self.sf = spk.representation.SymmetryFunctions(n_radial=radical, n_angular=angular, zetas=[1, 4],
                                                       elements=frozenset((26,)))
        self.L1 = nn.Linear(self.in_size, self.inter_size)
        self.L2 = nn.Linear(self.inter_size, self.out_size)
        self.L3 = nn.Linear(self.out_size, self.inter_size)
        self.L4 = nn.Linear(self.inter_size, self.in_size)

    def forward(self, X):
        X = F.leaky_relu(self.L1(X))
        a = F.sigmoid(self.L2(X))
        X = F.leaky_relu(self.L3(a))
        X = F.tanh(self.L4(X))
        return a, X

    # input为一个batch数据,该方法仅测试用
    def add_data(self, inputs):
        em = self.em(inputs[Properties.Z])
        acsf = self.sf(inputs)
        site = torch.cat([em, acsf], dim=2)
        inputs["_site"] = site
        return inputs