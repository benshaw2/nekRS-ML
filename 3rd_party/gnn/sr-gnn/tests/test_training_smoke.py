# tests/test_training_smoke.py

import torch
import types
import pytest

from torch_geometric.data import Data

from equivariance.packing import pack_flat_nekrs
from equivariance.generators import (
    build_node_generators_velocity,
    build_edge_generators_rotations,
    build_output_generators_velocity,
)
from equivariance.model_wrapper import make_model_flat
from equivariance.gamma_schedule import make_gamma_schedule
from symdisc.enforcement.regularization import (
    forward_with_equivariance_penalty,
    sum_fields,
    lift_field_to_flat_segment,
    as_field_lastdim,
)


class DummyModel(torch.nn.Module):
    """
    Simple nodewise equivariant model: y = ||u|| * u
    """
    def forward(
        self,
        x,
        mask=None,
        edge_index_lo=None,
        edge_index_hi=None,
        pos_lo=None,
        pos_hi=None,
        batch_lo=None,
        batch_hi=None,
        edge_index_coin=None,
        degree=None,
    ):
        #return x.norm(dim=1, keepdim=True) * x
        return x


def test_equivariant_training_smoke():
    torch.manual_seed(0)

    # ---- build dummy graph ----
    N = 6
    E = 8

    edge_index = torch.randint(0, N, (2, E))
    pos = torch.randn(N, 3)
    x = torch.randn(N, 3)

    data = Data(
        x=x,
        y=x.clone(),  # target irrelevant here
        pos_norm_lo=pos,
        pos_norm_hi=pos,
        edge_index_lo=edge_index,
        edge_index_hi=edge_index,
        x_batch=torch.zeros(N, dtype=torch.long),
        y_batch=torch.zeros(N, dtype=torch.long),
        central_element_mask=torch.ones(N, dtype=torch.bool),
        x_mean_lo=torch.zeros_like(x),
        x_std_lo=torch.ones_like(x),
        x_mean_hi=torch.zeros_like(x),
        x_std_hi=torch.ones_like(x),
        node_weight=torch.ones(N, 1),
    )

    # ---- model ----
    model = DummyModel()

    # ---- equivariance plumbing ----
    x_flat, meta = pack_flat_nekrs(data)
    offsets = meta["offsets"]


    X_nodes_base = build_node_generators_velocity()
    #X_nodes_base, names = build_node_generators_velocity(with_names=True)
    #rot_indices = [i for i, n in enumerate(names) if "R" in n]
    #X_nodes_base = [X_nodes_base[i] for i in rot_indices]

    X_edges_rot = build_edge_generators_rotations(
        num_edges=E,
        dx_offset=offsets["dx"],
        du_offset=offsets["du"],
    )

    X_nodes = [
        lift_field_to_flat_segment(
            as_field_lastdim(f, d=3),
            count=N,
            dim=3,
            offset=offsets["u"],
        )
        for f in X_nodes_base
    ]

    X_in = []
    for i in range(3):
        X_in.append(sum_fields(X_nodes[i], X_edges_rot[i]))
    for i in range(3, 6):
        X_in.append(X_nodes[i])

    Y_nodes = [
        lift_field_to_flat_segment(
            as_field_lastdim(f, d=3),
            count=N,
            dim=3,
            offset=offsets["u"],
        )
        for f in build_output_generators_velocity()
    ]

    Y_out = Y_nodes


    '''def zero_field(x, *, meta=None, grad=None):
        return torch.zeros_like(x)

    Y_out = Y_nodes + [zero_field] * len(X_edges_rot)'''


    model_flat = make_model_flat(
        model=model,
        data=data,
        device=torch.device("cpu"),
    )

    # ---- symmetry penalty ----
    '''y_flat, sym_pen = forward_with_equivariance_penalty(
        model=lambda xf: model_flat(xf, meta),
        X_in=X_in,
        Y_out=Y_out,
        x=x_flat,
    )'''
    y_flat, sym_pen = forward_with_equivariance_penalty(
        model=lambda xf: model_flat(xf[0], meta).unsqueeze(0),
        X_in=X_in,
        Y_out=Y_out,
        x=x_flat.unsqueeze(0),   # <-- CRITICAL
    )

    # For an equivariant model, penalty should be small
    assert sym_pen.item() < 1e-6