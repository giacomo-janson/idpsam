import torch
import numpy as np


def calc_dmap(xyz, epsilon=1e-12, backend="torch"):
    if backend == "torch":
        B = torch
    elif backend == "numpy":
        B = np
    else:
        raise KeyError(backend)
    if len(xyz.shape) == 2:
        if xyz.shape[1] != 3:
            raise ValueError(xyz.shape)
    elif len(xyz.shape) == 3:
        if xyz.shape[2] != 3:
            raise ValueError(xyz.shape)
    else:
        raise ValueError(xyz.shape)
    if len(xyz.shape) == 3:
        dmap = B.sqrt(
                 B.sum(
                   B.square(xyz[:,None,:,:] - xyz[:,:,None,:]),
                 axis=3) + epsilon)
        exp_dim = 1
    else:
        dmap = B.sqrt(
                 B.sum(
                   B.square(xyz[None,:,:] - xyz[:,None,:]),
                 axis=2) + epsilon)
        exp_dim = 0
    if backend == "torch":
        return dmap.unsqueeze(exp_dim)
    elif backend == "numpy":
        return np.expand_dims(dmap, exp_dim)
    else:
        raise KeyError(backend)


def calc_dmap_triu(input_data, epsilon=1e-12, backend="torch"):
    # Check the shape.
    if len(input_data.shape) == 2:
        if input_data.shape[1] != 3:
            raise ValueError(input_data.shape)
        dmap = calc_dmap(input_data, epsilon, backend)
    elif len(input_data.shape) == 3:
        if input_data.shape[2] != 3:
            raise ValueError(input_data.shape)
        dmap = calc_dmap(input_data, epsilon, backend)
    elif len(input_data.shape) == 4:
        if input_data.shape[1] != 1:
            raise ValueError(input_data.shape)
        if input_data.shape[2] != input_data.shape[3]:
            raise ValueError(input_data.shape)
        dmap = input_data
    else:
        raise ValueError(input_data.shape)
    # Get the triu ids.
    l = dmap.shape[2]
    if backend == "torch":
        triu_ids = torch.triu_indices(l, l, offset=1)
    elif backend == "numpy":
        triu_ids = np.triu_indices(l, k=1)
    else:
        raise KeyError(backend)
    # Returns the values.
    if len(input_data.shape) != 2:
        return dmap[:,0,triu_ids[0],triu_ids[1]]
    else:
        return dmap[0,triu_ids[0],triu_ids[1]]


def torch_chain_dihedrals(xyz, norm=False, backend="torch"):
    if backend == "torch":
        r_sel = xyz
    elif backend == "numpy":
        r_sel = torch.tensor(xyz)
    else:
        raise KeyError(backend)
    b0 = -(r_sel[:,1:-2,:] - r_sel[:,0:-3,:])
    b1 = r_sel[:,2:-1,:] - r_sel[:,1:-2,:]
    b2 = r_sel[:,3:,:] - r_sel[:,2:-1,:]
    b0xb1 = torch.cross(b0, b1)
    b1xb2 = torch.cross(b2, b1)
    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2)
    y = torch.sum(b0xb1_x_b1xb2*b1, axis=2)*(1.0/torch.linalg.norm(b1, dim=2))
    x = torch.sum(b0xb1*b1xb2, axis=2)
    dh_vals = torch.atan2(y, x)
    if not norm:
        return dh_vals
    else:
        return dh_vals/np.pi


def calc_chain_bond_angles(xyz):
    ids = np.array([[i, i+1, i+2] for i in range(xyz.shape[1]-2)])
    return calc_angles(xyz, ids)


def calc_angles(xyz, angle_indices):
    ix01 = angle_indices[:, [1, 0]]
    ix21 = angle_indices[:, [1, 2]]

    u_prime = xyz[:,ix01[:,1]]-xyz[:,ix01[:,0]]
    v_prime = xyz[:,ix21[:,1]]-xyz[:,ix01[:,0]]
    u_norm = np.sqrt((u_prime**2).sum(-1))
    v_norm = np.sqrt((v_prime**2).sum(-1))

    # adding a new axis makes sure that broasting rules kick in on the third
    # dimension
    u = u_prime / (u_norm[..., np.newaxis])
    v = v_prime / (v_norm[..., np.newaxis])

    return np.arccos((u * v).sum(-1))


# def compute_rg(xyz):
#     """
#     Adapted from the mdtraj library: https://github.com/mdtraj/mdtraj.
#     """
#     num_atoms = xyz.shape[1]
#     masses = np.ones(num_atoms)
#     weights = masses / masses.sum()
#     mu = xyz.mean(1)
#     centered = (xyz.transpose((1, 0, 2)) - mu).transpose((1, 0, 2))
#     squared_dists = (centered ** 2).sum(2)
#     Rg = (squared_dists * weights).sum(1) ** 0.5
#     return Rg


def sample_data(data, n_samples, backend="numpy"):
    if backend in ("numpy", "torch"):
        if n_samples is not None:
            ids = np.random.choice(data.shape[0],
                                   n_samples,
                                   replace=data.shape[0] < n_samples)
            return data[ids]
        else:
            return data
    else:
        raise KeyError(backend)


def get_edge_dmap(xyz, batch, epsilon=1e-12):
    row, col = batch.nr_edge_index
    dmap = torch.sqrt(
             torch.sum(
               torch.square(xyz[row] - xyz[col]),
             axis=1) + epsilon)
    return dmap