# equivariance/model_wrapper.py

import torch
import torch_geometric.nn as tgnn
from .packing import unpack_flat_nekrs


def make_model_flat(
    model,
    data,
    *,
    device,
    eps: float = 1e-10,
):
    """
    Create a flat-space wrapper around the nekRS-ML model.

    Parameters
    ----------
    model : torch.nn.Module
        The GNN model (already on device).
    data : torch_geometric.data.Data
        Batch currently being processed.
    device : torch.device
        CUDA / XPU / CPU.
    eps : float
        Numerical epsilon for scaling.

    Returns
    -------
    model_flat : callable
        Function mapping x_flat -> y_flat
    """

    # cache quantities used in closure
    # don't do this: #model.eval()  # symmetry penalties are differentiable; this just disables dropout

    # pull out things the model needs but are not in x_flat
    edge_index_lo = data.edge_index_lo
    edge_index_hi = data.edge_index_hi
    pos_lo = data.pos_norm_lo
    pos_hi = data.pos_norm_hi
    batch_lo = data.x_batch
    batch_hi = data.y_batch
    mask = data.central_element_mask

    x_mean_lo = data.x_mean_lo
    x_std_lo = data.x_std_lo
    x_std_hi = data.x_std_hi

    edge_index_coin = getattr(data, "edge_index_coin", None)
    degree = getattr(data, "degree", None)

    def model_flat(x_flat, meta):
        """
        Flat-space forward:
            x_flat -> unpack -> model -> flatten output
        """
        # ---- unpack primitive physical quantities ----
        u_nodes, _, _, _ = unpack_flat_nekrs(x_flat, meta)

        # ---- scale node velocities exactly as trainer does ----
        x_scaled = (u_nodes - x_mean_lo) / (x_std_lo + eps)

        # ---- forward through the original model ----
        out = model(
            x=x_scaled,
            mask=mask,
            edge_index_lo=edge_index_lo,
            edge_index_hi=edge_index_hi,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            batch_lo=batch_lo,
            batch_hi=batch_hi,
            edge_index_coin=edge_index_coin,
            degree=degree,
        )

        # ---- flatten outputs (fine-grid velocities) ----
        return out.reshape(-1)

    return model_flat
    
