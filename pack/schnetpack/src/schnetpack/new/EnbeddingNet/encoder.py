import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk
from schnetpack import Properties
import random


__all__ = ["Encoder"]


class Encoder(torch.nn.Module):
    def __init__(self, in_Size=128, inter1=96, out_Size=64,
                 em_len=64, radical=48, angular=4):
        super(Encoder, self).__init__()
        self.in_size = in_Size
        self.inter1_size = inter1
        self.out_size = out_Size
        self.em = nn.Embedding(100, em_len, padding_idx=0)
        self.sf = spk.representation.SymmetryFunctions(n_radial=radical, n_angular=angular, zetas=[1, 4],
                                                       elements=frozenset((1, 6, 8)))
        self.L1 = nn.Linear(self.in_size, self.inter1_size)
        self.L2 = nn.Linear(self.inter1_size, self.out_size)

    def forward(self, X):
        X = X.reshape(-1, self.in_size)
        X = F.relu(self.L1(X))
        X = F.relu(self.L2(X))
        return X

    # input为一个batch数据,该方法仅测试用
    def add_data(self, inputs):
        em = self.em(inputs[Properties.Z])
        acsf = self.sf(inputs)
        site = torch.cat([em, acsf], dim=2)
        inputs["_site"] = site
        return inputs

    # 使用三元损失函数时的数据生成函数
    def data_generator(self, data, triplet_batch=100):
        anchor = []
        pos = []
        nag = []
        atom_class = [1, 6, 8]
        atom_label = data[Properties.Z].reshape(-1)
        atom_label = atom_label.tolist()
        site_res = data["_site"].reshape(-1, 128)
        site_res = site_res.tolist()
        for i in range(triplet_batch):
            label = random.sample(atom_class, 2)

            anchor_num = random.sample(atom_label, 1)[0]
            while anchor_num != label[0]:
                anchor_num = random.sample(atom_label, 1)[0]
            anchor.append(site_res[anchor_num])

            pos_num = random.sample(atom_label, 1)[0]
            while pos_num != label[0]:
                pos_num = random.sample(atom_label, 1)[0]
            pos.append(site_res[pos_num])

            nag_num = random.sample(atom_label, 1)[0]
            while nag_num != label[1]:
                nag_num = random.sample(atom_label, 1)[0]
            nag.append(site_res[nag_num])
        return torch.tensor([anchor, pos, nag], dtype=torch.float32)