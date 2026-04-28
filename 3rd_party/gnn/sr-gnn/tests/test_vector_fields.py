# tests/test_vector_fields.py

import torch
import pytest

from equivariance.generators import (
    build_node_generators_velocity,
    build_edge_generators_rotations,
)
from equivariance.packing import compute_flat_offsets
from symdisc.enforcement.regularization import lift_field_to_flat_segment


def test_node_diagonal_action():
    """
    Each node should see the same infinitesimal action
    (diagonal SO(3) or Galilean generator).
    """
    torch.manual_seed(0)

    N = 4
    u = torch.randn(N, 3)
    x_flat = u.reshape(-1)

    X_nodes, names = build_node_generators_velocity(with_names=True)
    rot_idx = next(i for i, n in enumerate(names) if "R" in n)
    X = X_nodes[rot_idx] #[0]  # pick one generator

    offsets = compute_flat_offsets(num_nodes=N, num_edges=0)

    X_flat = lift_field_to_flat_segment(
        X,
        count=N,
        dim=3,
        offset=offsets["u"],
    )

    delta = X_flat(x_flat.unsqueeze(0))[0].reshape(N, 3)

    # All rows should differ only by the input u,
    # but the rule applied is identical.
    # In particular, zero input -> zero output.
    zero = torch.zeros_like(u)
    zero_flat = zero.reshape(-1)
    delta_zero = X_flat(zero_flat.unsqueeze(0))[0].reshape(N, 3)

    assert torch.allclose(delta_zero, torch.zeros_like(delta_zero), atol=1e-8)


def test_edge_scalar_invariance():
    """
    Rotation generators must NOT act on the ||Δx|| scalar slot.
    """
    torch.manual_seed(0)

    E = 5
    offsets = compute_flat_offsets(num_nodes=0, num_edges=E)

    x_flat = torch.randn(offsets["total_dim"])

    R_edges = build_edge_generators_rotations(
        num_edges=E,
        dx_offset=offsets["dx"],
        du_offset=offsets["du"],
    )

    # apply first rotation generator
    delta = R_edges[0](x_flat.unsqueeze(0))[0]

    scalar_block = delta[offsets["dx_norm"] : offsets["dx_norm"] + E]

    assert torch.allclose(
        scalar_block, torch.zeros_like(scalar_block), atol=1e-8
    ), "Scalar edge feature transformed under rotation!"