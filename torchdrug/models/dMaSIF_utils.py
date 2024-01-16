import torch
import torch.nn.functional as F
import torch.nn as nn
from math import sqrt
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster


# batch 内容就是每个原子对应的哪一个图结构，应该能直接使用
def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def subsample(x, batch=None, scale=1.0):
    """Subsamples the point cloud using a grid (cubic) clustering scheme.

    The function returns one average sample per cell, as described in Fig. 3.e)
    of the paper.

    Args:
        x (Tensor): (N,3) point cloud.
        batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric.
            Defaults to None.
        scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).

    Returns:
        (M,3): sub-sampled point cloud, with M <= N.
    """

    if batch is None:  # Single protein case:
        labels = grid_cluster(x, scale).long()
        C = labels.max() + 1
        # We append a "1" to the input vectors, in order to
        # compute both the numerator and denominator of the "average"
        # fraction in one pass through the data.
        x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        D = x_1.shape[1]
        points = torch.zeros_like(x_1[:C])
        points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
        return (points[:, :-1] / points[:, -1:]).contiguous()

    else:  # We process proteins using a for loop.
        # This is probably sub-optimal, but I don't really know
        # how to do more elegantly (this type of computation is
        # not super well supported by PyTorch).
        batch_size = torch.max(batch).item() + 1  # Typically, =32
        points, batches = [], []
        for b in range(batch_size):
            p = subsample(x[batch == b], scale=scale)
            points.append(p)
            batches.append(b * torch.ones_like(batch[: len(p)]))
        return torch.cat(points, dim=0), torch.cat(batches, dim=0)


def soft_distances(x, y, batch_x, batch_y, atomtypes, smoothness=0.01):
    """Computes a soft distance function to the atom centers of a protein.

    Implements Eq. (1) of the paper in a fast and numerically stable way.

    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
        batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
        smoothness (float, optional): atom radii if atom types are not provided. Defaults to .01.
        atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.

    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    # Build the (N, M, 1) symbolic matrix of squared distances:
    x_i = LazyTensor(x[:, None, :])  # (N, 1, 3) atoms
    y_j = LazyTensor(y[None, :, :])  # (1, M, 3) sampling points
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1) squared distances

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    D_ij.ranges = diagonal_ranges(batch_x, batch_y)
    # Turn the one-hot encoding "atomtypes" into a vector of diameters "smoothness_i":
    # atomic_radii = torch.cuda.FloatTensor(
    #     [1.10, 1.92, 1.70, 1.55, 1.52, 1.47, 1.73, 2.10, 1.80, 1.80,
    #      1.75, 1.40, 1.39, 1.90, 1.85, 2.17, 1.98], device=x.device
    # )
    atomic_radii = torch.cuda.FloatTensor(
        [1.46, 1.54, 1.64, 1.88, 1.99, 1.61, 1.80, 1.99, 2.02, 1.83,
         1.88, 1.96, 2.17, 2.07, 2.05, 2.02, 2.18, 2.17, 2.19, 2.38], device=x.device
    )
    atomic_radii = atomic_radii / atomic_radii.min()
    atomtype_radii = atomtypes * atomic_radii[None, :]  # n_atoms, n_atomtypes
    # smoothness = atomtypes @ atomic_radii  # (N, 6) @ (6,) = (N,)
    smoothness = torch.sum(smoothness * atomtype_radii, dim=1, keepdim=False)  # n_atoms, 1
    smoothness_i = LazyTensor(smoothness[:, None, None])
    # Compute an estimation of the mean smoothness in a neighborhood of each sampling point:
    mean_smoothness = (-D_ij.sqrt()).exp().sum(0)
    mean_smoothness_j = LazyTensor(mean_smoothness[None, :, :])
    mean_smoothness = (smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j)  # n_atoms, n_points, 1
    mean_smoothness = mean_smoothness.sum(0).view(-1)
    soft_dists = - mean_smoothness * ((-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)).view(-1)
    return soft_dists


def atoms_to_points_normals(
        atom_coords, batch, atomtypes, distance=1.05, smoothness=0.5, resolution=1.0, nits=4, sup_sampling=10, variance=0.1,
):
    """Turns a collection of atoms into an oriented point cloud.

    Sampling algorithm for protein surfaces, described in Fig. 3 of the paper.

    Args:
        atoms (Tensor): (N,3) coordinates of the atom centers `a_k`.
        batch (integer Tensor): (N,) batch vector, as in PyTorch_geometric.
        distance (float, optional): value of the level set to sample from
            the smooth distance function. Defaults to 1.05.
        smoothness (float, optional): radii of the atoms, if atom types are
            not provided. Defaults to 0.5.
        resolution (float, optional): side length of the cubic cells in
            the final sub-sampling pass. Defaults to 1.0.
        nits (int, optional): number of iterations . Defaults to 4.
        atomtypes (Tensor, optional): (N,6) one-hot encoding of the atom
            chemical types. Defaults to None.

    Returns:
        (Tensor): (M,3) coordinates for the surface points `x_i`.
        (Tensor): (M,3) unit normals `n_i`.
        (integer Tensor): (M,) batch vector, as in PyTorch_geometric.
    """
    # a) Parameters for the soft distance function and its level set:
    T = distance
    N, D = atom_coords.shape
    B = sup_sampling  # Sup-sampling ratio
    # Batch vectors:
    batch_atoms = batch
    batch_z = batch[:, None].repeat(1, B).view(N * B)
    # b) Draw N*B points at random in the neighborhood of our atoms
    z = atom_coords[:, None, :] + 10 * T * torch.randn(N, B, D).type_as(atom_coords)
    z = z.view(-1, D)  # (N*B, D)
    # We don't want to backprop through a full network here!
    atoms = atom_coords.detach().contiguous()
    z = z.detach().contiguous()
    # N.B.: Test mode disables the autograd engine: we must switch it on explicitely.
    with torch.enable_grad():
        if z.is_leaf:
            z.requires_grad = True
        # c) Iterative loop: gradient descent along the potential
        # ".5 * (dist - T)^2" with respect to the positions z of our samples
        for it in range(nits):
            dists = soft_distances(atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes)
            Loss = ((dists - T) ** 2).sum()
            g = torch.autograd.grad(Loss, z)[0]
            # g = torch.where(torch.isnan(g), torch.full_like(g, 0), g)  # add for nan
            z.data -= 0.5 * g

        # d) Only keep the points which are reasonably close to the level set:
        dists = soft_distances(atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes)
        margin = (dists - T).abs()
        mask = margin < variance * T

        # d') And remove the points that are trapped *inside* the protein:
        zz = z.detach()
        zz.requires_grad = True
        for it in range(nits):
            dists = soft_distances(atoms, zz, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes)
            Loss = (1.0 * dists).sum()
            g = torch.autograd.grad(Loss, zz)[0]
            # g = torch.where(torch.isnan(g), torch.full_like(g, 0), g)  # add for nan
            normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
            zz = zz + 1.0 * T * normals

        dists = soft_distances(atoms, zz, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes)
        mask = mask & (dists > 1.5 * T)
        z = z[mask].contiguous().detach()
        batch_z = batch_z[mask].contiguous().detach()

        # e) Subsample the point cloud:
        points, batch_points = subsample(z, batch_z, scale=resolution)

        # f) Compute the normals on this smaller point cloud:
        p = points.detach()
        p.requires_grad = True
        dists = soft_distances(atoms, p, batch_atoms, batch_points, smoothness=smoothness, atomtypes=atomtypes)
        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss, p)[0]
        # g = torch.where(torch.isnan(g), torch.full_like(g, 0), g)  # add for nan
        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
    points = points - 0.5 * normals
    return points.detach(), normals.detach(), batch_points.detach()


def tangent_vectors(normals):
    """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].

          normals        ->             uv
    (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)

    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".

    Args:
        normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

    Returns:
        (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
            the tangent coordinate systems `[n_i,u_i,v_i].
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv


def knn_atoms(x, y, x_batch, y_batch, k):
    N, D = x.shape  # [N_points, 3]
    x_i = LazyTensor(x[:, None, :])  # [N_points, 1, 3]
    y_j = LazyTensor(y[None, :, :])  # [1, N_atoms, 3]

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)  # [N_points, N_atoms]
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # [N_points, 16]
    x_ik = y[idx.view(-1)].view(N, k, D) # [N_points, 16, 3]
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)  # [N_points, 16]

    return idx, dists

def get_atom_features(point_coords, atom_coords, point_batch, atom_batch, atom_feature, k=16):
    idx, dists = knn_atoms(point_coords, atom_coords, point_batch, atom_batch, k=k)  # [N_points, k]
    num_points, _ = idx.size()

    idx = idx.view(-1)  # [N_points * k,]
    dists = 1 / dists.view(-1, 1)  # [N_points * k , 1]
    _, num_dims = atom_feature.size()  # 17

    feature = atom_feature[idx, :]  # [N_points * k, 17]
    feature = torch.cat([feature, dists], dim=1)  # [N_points * k, 18]
    feature = feature.view(num_points, k, num_dims + 1)  # [N_points, k, 17]

    return feature, idx