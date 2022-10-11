import torch
import torch.nn as nn

from schnetpack.nn.base import Dense
from schnetpack import Properties
from schnetpack.nn.cfconv import CFConv
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.acsf import GaussianSmearing
from schnetpack.nn.neighbors import AtomDistances
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.cfconv import Attention
from schnetpack.representation.hdnn import SymmetryFunctions
from ase import Atoms
from ase.neighborlist import neighbor_list


__all__ = ["SchNetInteraction", "SchNet"]


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        n_filters (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_filters,
        cutoff,
        cutoff_network=CosineCutoff,
        normalize_filter=False,
    ):
        super(SchNetInteraction, self).__init__()
        # filter block used in interaction block
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters),
        )
        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)
        # interaction block
        self.cfconv = CFConv(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=3,
        cutoff=5.0,
        n_gaussians=25,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=100,
        cutoff_network=CosineCutoff,
        trainable_gaussians=False,
        distance_expansion=None,
        charged_systems=False,
        Gass_true_atten_flase=True,
        pbc=False,
    ):
        super(SchNet, self).__init__()

        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size n_atom_basis
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # layer for computing interatomic distances
        self.distances = AtomDistances()

        # Attention or Gassuian
        self.G_or_A = Gass_true_atten_flase

        # pbc
        self.pbc = pbc

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion = distance_expansion

        # block for computing interaction
        if self.G_or_A:
            if coupled_interactions:
                # use the same SchNetInteraction instance (hence the same weights)
                self.interactions = nn.ModuleList(
                    [
                        SchNetInteraction(
                            n_atom_basis=n_atom_basis,
                            n_spatial_basis=n_gaussians,
                            n_filters=n_filters,
                            cutoff_network=cutoff_network,
                            cutoff=cutoff,
                            normalize_filter=normalize_filter,
                        )
                    ]
                    * n_interactions
                )
            else:
                # use one SchNetInteraction instance for each interaction
                self.interactions = nn.ModuleList(
                    [
                        SchNetInteraction(
                            n_atom_basis=n_atom_basis,
                            n_spatial_basis=n_gaussians,
                            n_filters=n_filters,
                            cutoff_network=cutoff_network,
                            cutoff=cutoff,
                            normalize_filter=normalize_filter,
                        )
                        for _ in range(n_interactions)
                    ]
                )
        else:
            self.interactions = nn.ModuleList(
                [
                    Attention(
                        n_in=n_atom_basis,
                        n_out=n_atom_basis,
                        dropout=0.3,
                        alpha=0.2,
                        cut_off=cutoff
                    )
                    for _ in range(n_interactions)
                ]
            )
            self.old_sch = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

    def forward(self, inputs):
        """Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)
        # Newly added variables, built using EmbeddingNet module
        # x = inputs[Properties.new_embedding]

        if self.G_or_A:
            if self.charged_systems and Properties.charge in inputs.keys():
                n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
                charge = inputs[Properties.charge] / n_atoms  # B
                charge = charge[:, None] * self.charge  # B x F
                x = x + charge

            # compute interatomic distance of every atom to its neighbors
            if self.pbc:
                r_ij = self._cal_pbc(positions, neighbors, inputs[Properties.pbc_offset], inputs[Properties.cell])
            else:
                r_ij = self.distances(
                    positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
                )
            # expand interatomic distances (for example, Gaussian smearing)
            f_ij = self.distance_expansion(r_ij)
            # store intermediate representations
            if self.return_intermediate:
                xs = [x]
            # compute interaction block to update atomic embeddings
            for interaction in self.interactions:
                v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
                x = x + v
                if self.return_intermediate:
                    xs.append(x)
        else:
            if self.charged_systems and Properties.charge in inputs.keys():
                n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
                charge = inputs[Properties.charge] / n_atoms  # B
                charge = charge[:, None] * self.charge  # B x F
                x = x + charge

            # compute interatomic distance of every atom to its neighbors
            r_ij = self.distances(
                positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
            )
            # expand interatomic distances (for example, Gaussian smearing)
            f_ij = self.distance_expansion(r_ij)
            # store intermediate representations
            if self.return_intermediate:
                xs = [x]
            # compute interaction block to update atomic embeddings
            for interaction, old_sh in zip(self.interactions, self.old_sch):
                v = old_sh(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
                x = interaction(v, positions)
                if self.return_intermediate:
                    xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x

    def _cal_pbc(self, position, neighbors, offset, cell):
        batch_size = position.size()[0]
        idx_m = torch.arange(batch_size, device=position.device, dtype=torch.long)[
                :, None, None
                ]
        # Get atomic positions of all neighboring indices
        pos_xyz = position[idx_m, neighbors[:, :, :], :]
        dist_vec = pos_xyz - position[:, :, None, :]

        # offset = torch.zeros_like(pos_xyz, device=position.device)
        # pos = torch.clone(position).detach().cpu()
        # for i in range(batch_size):
        #     pos_ = pos[i].squeeze(0).tolist()
        #     atoms = Atoms(positions=pos_, cell=[cell, cell, cell], pbc=True)
        #     nh1, nh2, s = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
        #     for j in range(len(nh1)):
        #         offset[i, nh1[j], nh2[j] - 1, :] = torch.tensor(s[j])
        cell_len = cell.size()[0]
        res = torch.tensor([], dtype=torch.float32, device='cuda')
        c = torch.clone(cell).tolist()
        for i in range(cell_len):
            temp = []
            temp.append(c[i][0][0])
            temp.append(c[i][1][1])
            temp.append(c[i][2][2])
            temp = torch.tensor(temp, device='cuda')
            temp = offset[i] * temp
            res = torch.cat((res, temp.unsqueeze(0)), dim=0)

        #这里有问题
        dis = dist_vec + res
        dis = torch.norm(dis, 2, dim=-1)
        return dis

