import torch.nn as nn
import torch
import torch.nn.functional as F

# in_feature = 8
# out_feature = 4
# alpha = 0.2
# dropout = 0.3
# W = nn.Parameter(torch.empty(size=(in_feature, out_feature)))
# nn.init.xavier_uniform_(W.data, gain=1.414)
# a = nn.Parameter(torch.empty(size=(2 * out_feature, 1)))
# nn.init.xavier_uniform_(a.data, gain=1.414)
# Lelu = nn.LeakyReLU(alpha)
# Dropout = nn.Dropout(dropout)
#
# def prepare_attentional_mechanism_input(Wh):
#     N = Wh.size()[1]
#     Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # 此处操作是将1维上张量复制N遍，最终效果1,1,1..n,n,n
#     Wh_repeated_alternating = Wh.repeat(1, N, 1)  # 此处操作是将1维上所有张量复制N遍，最终效果1,2,..n,1,2..n,..
#     all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
#     return all_combinations_matrix.view(-1, N, N, 2 * out_feature)
#
# x = torch.randn([100, 9, 8])
# xx = torch.randn([100, 9, 3])
# x_temp1 = xx.repeat_interleave(9, dim=1)
# x_temp2 = xx.repeat(1, 9, 1)
# res = x_temp1 - x_temp2
# res = torch.norm(res, 2, dim=2)
# res = res.reshape(-1, 9, 9)
# print(res)
#
#
# Wh = torch.matmul(x, W)  # x.shape: (N_b, N_a, N_in), W.shape: (N_in, N_out)
# # -> Wh.shape: (N_b, N_a, N_out)
# a_input = prepare_attentional_mechanism_input(Wh)  # a_input.shape: (N_b, N_a, N_a, 2*N_out)
# e = Lelu(torch.matmul(a_input, a).squeeze(3))  # e.shape: (N_b, N_a, N_a)
# zero_vec = -9e15 * torch.ones_like(e)
# attention = torch.where(res > 2., zero_vec, e)
# attention = F.softmax(attention, dim=2)
# attention = Dropout(attention)
# h_prime = torch.matmul(attention, Wh)  # h_prime.shape: (N_b, N_a, N_out)
# print(h_prime.shape)

a = torch.zeros([100, 128, 127, 3])
a[0, 0, 0, :] = torch.tensor([1, -1, 1])
print(a)
