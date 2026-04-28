# equivariance/packing.py

import torch


# ------------------------------------------------------------
# Offset bookkeeping
# ------------------------------------------------------------

def compute_flat_offsets(num_nodes: int, num_edges: int):
    """
    Compute offsets for the canonical flat representation:

      x_flat =
        [ u_nodes (N×3),
          Δx_edges (E×3),
          |Δx_edges| (E×1),
          Δu_edges (E×3) ]

    Returns a dict of offsets and total dimension.
    """

    offsets = {}

    offset = 0

    offsets["u"] = offset
    offset += num_nodes * 3

    offsets["dx"] = offset
    offset += num_edges * 3

    offsets["dx_norm"] = offset
    offset += num_edges * 1

    offsets["du"] = offset
    offset += num_edges * 3

    offsets["total_dim"] = offset

    return offsets


# ------------------------------------------------------------
# Packing
# ------------------------------------------------------------

def pack_flat_nekrs(data):
    """
    Pack nekRS-ML physical data into a flat vector suitable for
    equivariance enforcement.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Must contain:
          - data.x               : (N, 3) node velocities
          - data.pos_norm_lo     : (N, 3) node positions
          - data.edge_index_lo   : (2, E) edge indices

    Returns
    -------
    x_flat : torch.Tensor, shape (D,)
    meta   : dict
        Contains offsets and sizes needed to unpack.
    """

    device = data.x.device
    dtype = data.x.dtype

    # ---- basic sizes ----
    N = data.x.shape[0]
    edge_index = data.edge_index_lo
    E = edge_index.shape[1]

    # ---- offsets ----
    offsets = compute_flat_offsets(N, E)

    # ---- node velocities ----
    u = data.x.reshape(-1)  # (N*3,)

    # ---- edge geometry ----
    src = edge_index[0]
    dst = edge_index[1]

    pos_i = data.pos_norm_lo[src]
    pos_j = data.pos_norm_lo[dst]

    dx = pos_i - pos_j                       # (E, 3)
    dx_norm = torch.norm(dx, dim=1, keepdim=True)  # (E, 1)

    # ---- edge velocity differences ----
    u_i = data.x[src]
    u_j = data.x[dst]
    du = u_i - u_j                           # (E, 3)

    # ---- flatten everything ----
    x_flat = torch.empty(offsets["total_dim"], device=device, dtype=dtype)

    x_flat[offsets["u"]:offsets["u"] + N*3] = u
    x_flat[offsets["dx"]:offsets["dx"] + E*3] = dx.reshape(-1)
    x_flat[offsets["dx_norm"]:offsets["dx_norm"] + E] = dx_norm.reshape(-1)
    x_flat[offsets["du"]:offsets["du"] + E*3] = du.reshape(-1)

    meta = {
        "num_nodes": N,
        "num_edges": E,
        "offsets": offsets,
    }

    return x_flat, meta


# ------------------------------------------------------------
# Unpacking
# ------------------------------------------------------------

def unpack_flat_nekrs(x_flat, meta):
    """
    Inverse of pack_flat_nekrs.

    Parameters
    ----------
    x_flat : torch.Tensor, shape (D,)
    meta   : dict
        Returned from pack_flat_nekrs

    Returns
    -------
    u_nodes : (N, 3)
    dx      : (E, 3)
    dx_norm : (E, 1)
    du      : (E, 3)
    """

    offsets = meta["offsets"]
    N = meta["num_nodes"]
    E = meta["num_edges"]

    # ---- node velocities ----
    u_start = offsets["u"]
    u_end   = u_start + N*3
    u_nodes = x_flat[u_start:u_end].reshape(N, 3)

    # ---- Δx ----
    dx_start = offsets["dx"]
    dx_end   = dx_start + E*3
    dx = x_flat[dx_start:dx_end].reshape(E, 3)

    # ---- ||Δx|| ----
    n_start = offsets["dx_norm"]
    n_end   = n_start + E
    dx_norm = x_flat[n_start:n_end].reshape(E, 1)

    # ---- Δu ----
    du_start = offsets["du"]
    du_end   = du_start + E*3
    du = x_flat[du_start:du_end].reshape(E, 3)

    return u_nodes, dx, dx_norm, du
