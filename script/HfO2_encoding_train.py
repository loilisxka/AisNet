import os
from schnetpack.datasets import MD17
import torch
from torch import optim
import schnetpack as spk
import csv
from encoder import Encoder
from schnetpack import Properties
import numpy as np
import random

ethanol = "../data/HfO2/HfO2.db"
model_path = "./HfO2_Triplet"
learning_rate = 0.001
batch = 50
# 三元组的batch
triplet_batch = 2000
#train_sample = 200000
#val_sample = 10000
train_sample = 0.8
val_sample = 0.1
epoch = 50
total_train_step = 0
total_test_step = 0

if not os.path.exists(model_path):
    os.makedirs(model_path)

# ethanol_data = MD17(os.path.join(ethanol, 'malonaldehyde.db'), molecule='malonaldehyde', collect_triples=True)
ethanol_data = AtomsData(ethanol, collect_triples=True)

print(ethanol_data[0][Properties.Z])

out = open(os.path.join(model_path, "result.csv"), "a")
csv_writer = csv.writer(out, dialect="excel")

# 划分数据集
# 重新设计数据比例时需要将split.npz删除!!!!
train, test, val = spk.train_test_split(
    data=ethanol_data,
    num_train=train_sample,
    num_val=val_sample,
    split_file=os.path.join(model_path, "split.npz")
)

# 载入数据
train_loader = spk.AtomsLoader(train, batch_size=batch, shuffle=True, num_workers=12, pin_memory=True)
test_loader = spk.AtomsLoader(test, batch_size=batch, shuffle=True, num_workers=6, pin_memory=True)
val_loader = spk.AtomsLoader(val, batch_size=batch, num_workers=6, pin_memory=True)

model = Encoder()
device = torch.device('cuda')
model.to(device)
# 打印网络结构
print(model)

# data为一个batch的数据, 默认data已经加入_site属性
# def data_generator(data):
#     anchor = []
#     pos = []
#     nag = []
#     atom_class = [1, 6, 8]
#     atom_label = data[Properties.Z].reshape(-1)
#     atom_label = atom_label.tolist()
#     site_res = data["_site"].reshape(-1, 128)
#     site_res = site_res.tolist()
#     for i in range(triplet_batch):
#         label = random.sample(atom_class, 2)
#
#         anchor_num = random.sample(atom_label, 1)[0]
#         while anchor_num != label[0]:
#             anchor_num = random.sample(atom_label, 1)[0]
#         anchor.append(site_res[anchor_num])
#
#         pos_num = random.sample(atom_label, 1)[0]
#         while pos_num != label[0]:
#             pos_num = random.sample(atom_label, 1)[0]
#         pos.append(site_res[pos_num])
#
#         nag_num = random.sample(atom_label, 1)[0]
#         while nag_num != label[1]:
#             nag_num = random.sample(atom_label, 1)[0]
#         nag.append(site_res[nag_num])
#     return torch.tensor([anchor, pos, nag], dtype=torch.float32)

# 在原有数据中加入初始化的节点信息,并生成三元组
# for data in train_loader:
#     data = model.add_data(data)
#     print(data["_site"].shape)
#     triplet_data = data_generator(data).reshape(-1, 128)
#     print(triplet_data.shape)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 冻结Embedding的训练
for name, param in model.named_parameters():
    if "embedding" in name:
        param.requires_grad = False

TL_loss = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(), reduction='sum', margin=1.).to(device)

#for p in model.parameters():
#    print(p.gard)
#    if p.grad is not None:
#        print(p.grad.data)

# 设置训练过程
best_test_loss = float('inf')
for i in range(epoch):
    print("----------第{}轮训练----------".format(i))
    total_train_loss = 0
    total_train_step = 0
    #params = list(model.named_parameters())
    #(name, param) = params[0]
    #print(name)
    #print(params)
    model.train()
    for inputs in train_loader:
        # 特征提取器的输出分别为，Embedding结果，自编码器中间结果，自编码器输出结果
        # inputs = model.add_data(inputs)
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = model.add_data(inputs)
        inputs['_site'] = inputs['_site'].to(device)
        anchor, pos, nag = model.data_generator(inputs, triplet_batch)
        #triplet_data = triplet_data.to(device)
        anchor1 = anchor.to(device)
        pos1 = pos.to(device)
        nag1 = nag.to(device)
        anchor = model(anchor1)
        pos = model(pos1)
        nag = model(nag1)
        loss = TL_loss(anchor, pos, nag)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()/total_train_step))

    # 测试步骤
    total_test_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # data = model.add_data(data)
            data = {k: v.to(device) for k, v in data.items()}
            data = model.add_data(data)
            data['_site'] = data['_site'].to(device)
            a1, p1, n1 = model.data_generator(data, triplet_batch)
            #triplet_data = triplet_data.to(device)
            a1 = a1.to(device)
            p1 = p1.to(device)
            n1 = n1.to(device)
            anchor = model(a1)
            pos = model(p1)
            nag = model(n1)
            tloss = TL_loss(anchor, pos, nag)
            total_test_loss += tloss.item()
            #print("测试集Loss为：{}".format(total_test_loss/triplet_batch))

    # # 保存最好模型
    # if best_test_loss >= total_test_loss:
    #     best_test_loss = total_test_loss
    torch.save(model, "Triplet_HfO2_e_64_a_64.pth")
    #model = model

    outcome = [i, total_train_loss/(triplet_batch * 10)]#, total_test_loss/(triplet_batch * 10)]
    csv_writer.writerow(outcome)

out.close()
