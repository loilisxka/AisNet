import torch
from torch import nn
import torch.nn.functional as F

from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate

__all__ = ["CFConv", "Attention"]


class CFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters (int): number of filter dimensions.
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
    ):
        super(CFConv, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W = self.filter_network(f_ij)
        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)
        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        return y


class Attention(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 dropout,
                 alpha,
                 cut_off,
                 chuandi=True):
        super(Attention, self).__init__()
        self.in_feature = n_in
        self.out_feature = n_out
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(self.in_feature, self.out_feature)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * self.out_feature, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.Lelu = nn.LeakyReLU(self.alpha)
        self.Dropout = nn.Dropout(self.dropout)
        self.Chuandi=chuandi
        self.cut_off = cut_off

    def forward(self, x, position):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            position (torch.Tensor): 输入的各原子坐标，形状为(N_b, N_a, 3)
            cut_off (float): 输入的截至半径

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.
        """
        Wh = torch.matmul(x, self.W)  # x.shape: (N_b, N_a, N_in), W.shape: (N_in, N_out)
        # -> Wh.shape: (N_b, N_a, N_out)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # a_input.shape: (N_b, N_a, N_a, 2*N_out)
        e = self.Lelu(torch.matmul(a_input, self.a).squeeze(3))  # e.shape: (N_b, N_a, N_a)
        zero_vec = -9e15 * torch.ones_like(e)
        dis = self._cal_distiance(position)
        attention = torch.where(dis > self.cut_off, zero_vec, e)
        attention = F.softmax(attention, dim=2)
        attention = self.Dropout(attention)
        h_prime = torch.matmul(attention, Wh)  # h_prime.shape: (N_b, N_a, N_out)
        if self.Chuandi:
            h_prime = F.elu(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # 此处操作是将1维上张量复制N遍，最终效果1,1,1..n,n,n
        Wh_repeated_alternating = Wh.repeat(1, N, 1)  # 此处操作是将1维上所有张量复制N遍，最终效果1,2,..n,1,2..n,..
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(-1, N, N, 2 * self.out_feature)

    def _cal_distiance(self, position):
        Atom = position.size()[1]
        x_temp1 = position.repeat_interleave(Atom, dim=1)
        x_temp2 = position.repeat(1, Atom, 1)
        res = x_temp1 - x_temp2
        res = torch.norm(res, 2, dim=2)
        return res.view(-1, Atom, Atom)
