from ase import Atoms
from ase.neighborlist import neighbor_list
import torch


DTYPE = torch.float32
DEVICE = 'cpu'
def pbc_edges(cutoff, z, x, cell, batch, compute_sc=False):
    NH1 = torch.tensor([], dtype=torch.long, device=DEVICE)
    NH2 = torch.tensor([], dtype=torch.long, device=DEVICE)
    S = torch.tensor([], dtype=torch.long, device=DEVICE)
    D = torch.tensor([], dtype=DTYPE, device=DEVICE)
    SC = torch.tensor([], dtype=DTYPE, device=DEVICE) if compute_sc else None
    x_ = torch.clone(x).detach().cpu().numpy()

    if batch is not None:
        # count number of elements per batch
        batch_ids = list(set(batch.cpu().tolist()))
        batch_sizes = [(batch == id).sum().item() for id in batch_ids]

        for i in range(len(batch_sizes)):
            offset = sum(batch_sizes[:i])  # to obtain correct atom indices

            atoms = Atoms(charges=(z[offset:offset + batch_sizes[i]]).cpu(),
                          positions=x_[offset:offset + batch_sizes[i]],
                          cell=(cell[3 * i:3 * (i + 1)]).cpu(),
                          pbc=True
                          )

            nh1, nh2, s = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
            nh1 = torch.tensor(nh1, dtype=torch.long, device=DEVICE)
            nh2 = torch.tensor(nh2, dtype=torch.long, device=DEVICE)
            nh1 = nh1 + offset
            nh2 = nh2 + offset
            s = torch.tensor(s, dtype=DTYPE, device=DEVICE)
            d = x[nh2] - x[nh1] + torch.matmul(s, cell[3 * i:3 * (i + 1)])

            if compute_sc:
                cell_flat = torch.flatten(cell[3 * i:3 * (i + 1)])
                sc = torch.tile(cell_flat, (len(d), 1))
                sc[:, 0:3] = (sc[:, 0:3].T * s[:, 0]).T
                sc[:, 3:6] = (sc[:, 3:6].T * s[:, 1]).T
                sc[:, 6:9] = (sc[:, 6:9].T * s[:, 2]).T
                SC = torch.cat((SC, sc), 0)

            NH1 = torch.cat((NH1, nh1), 0)
            NH2 = torch.cat((NH2, nh2), 0)
            S = torch.cat((S, s), 0)
            D = torch.cat((D, d), 0)

    else:  # no batch
        atoms = Atoms(charges=z.cpu(), positions=x.cpu(), cell=cell.cpu(), pbc=True)
        nh1, nh2, s = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
        nh1 = torch.tensor(nh1, dtype=torch.long, device=DEVICE)
        nh2 = torch.tensor(nh2, dtype=torch.long, device=DEVICE)
        s = torch.tensor(s, dtype=DTYPE, device=DEVICE)
        d = x[nh2] - x[nh1] + torch.matmul(s, cell)

        if compute_sc:
            cell_flat = torch.flatten(cell)
            sc = torch.tile(cell_flat, (len(d), 1))
            sc[:, 0:3] = (sc[:, 0:3].T * s[:, 0]).T
            sc[:, 3:6] = (sc[:, 3:6].T * s[:, 1]).T
            sc[:, 6:9] = (sc[:, 6:9].T * s[:, 2]).T
            SC = sc

        NH1, NH2, S, D = nh1, nh2, s, d

    D = D.norm(dim=-1)
    return NH1, NH2, D, S, SC
