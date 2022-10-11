import os
import torch
from torch import optim
import schnetpack as spk
import csv
from AutoEncoder import AutoEncoder
from schnetpack.datasets import AtomsData
from schnetpack.datasets import MD17
from schnetpack import Properties
from torch import nn

fetut = "./Cu_e_64_a_64"
learning_rate = 0.01
batch = 8

epoch = 50
total_train_step = 0
total_test_step = 0

if not os.path.exists(fetut):
    os.makedirs(fetut)

new_dataset = AtomsData("../data/Cu/Cu.db", collect_triples=True)

out1 = open(os.path.join(fetut, "result.csv"), "a")
csv_writer = csv.writer(out1, dialect="excel")


# 准备数据
train, val, test = spk.train_test_split(
    data=new_dataset,
    num_train=0.8,
    num_val=0.1,
    split_file=os.path.join(fetut, "split.npz")
)

print(len(train))

train_loader = spk.AtomsLoader(train, batch_size=batch, shuffle=True, num_workers=16)
test_loader = spk.AtomsLoader(test, batch_size=batch, shuffle=True, num_workers=16)
val_loader = spk.AtomsLoader(val, batch_size=batch)

test_len = len(test_loader)
train_len = len(train_loader)

means, stddevs = train_loader.get_statistics(
    spk.datasets.MD17.energy, divide_by_atoms=True
)

print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(means[MD17.energy][0]))
print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(stddevs[MD17.energy][0]))

model = AutoEncoder()
device = torch.device('cuda')
model = model.to(device)
# 打印网络结构
print(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

LossFun = torch.nn.MSELoss().to(device)

# 冻结Embedding的训练
#for name, param in model.named_parameters():
#    if "embedding" in name:
#        param.requires_grad = False

# 设置训练过程
best_test_loss = float('inf')
for i in range(epoch):
    print("----------第{}轮训练----------".format(i))
    total_train_loss = 0
    total_train_step = 0
    for inputs in train_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        #print(inputs)
        inputs = model.add_data(inputs)
        inputs[Properties.new_embedding].to(device)
        atom_num = inputs[Properties.new_embedding].size()[1]
        x = inputs[Properties.new_embedding].view(-1, 128)
        res, out = model(x)
        loss = LossFun(x, out)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 20 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item() / (batch * 128)))

    # 测试步骤
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = {k: v.to(device) for k, v in data.items()}
            data = model.add_data(data)
            data[Properties.new_embedding].to(device)
            atom_num = data[Properties.new_embedding].size()[1]
            x = data[Properties.new_embedding].view(-1, 128)
            res, out = model(x)
            loss = LossFun(x, out)
            total_test_loss += loss.item()
    print("测试集Loss为：{}".format(total_test_loss/(batch * 128)))
    if best_test_loss >= total_test_loss:
        best_test_loss = total_test_loss
        torch.save(model, "Cu_AutoEncoder_e_64_a_64_bcc.pth")

    outcome = [i, total_train_loss / train_len, total_test_loss / test_len]
    csv_writer.writerow(outcome)

out1.close()
