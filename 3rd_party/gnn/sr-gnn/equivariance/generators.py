# equivariance/generators.py

import math
from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.enforcement.regularization import (
    as_field_lastdim,
    lift_field_to_flat_segment,
    sum_fields,
    pad_field,
)


def build_node_generators_velocity():
    """
    Node generators for velocity features u ∈ ℝ³.

    Includes:
      • 3 rotations (SO(3))
      • 3 Galilean boosts (translations in velocity space)
    """

    fields, names = generate_euclidean_killing_fields_with_names(
        d=3,
        include_rotations=True,
        include_translations=True,
        backend="torch",
    )

    # act on last dimension
    X_nodes = [as_field_lastdim(f, d=3) for f in fields]
    return X_nodes



def build_output_generators_velocity():
    """
    Output generators for predicted velocity u_fine ∈ ℝ³.

    Identical representation to node velocities.
    """
    return build_node_generators_velocity()


def build_edge_generators_rotations(
    *,
    num_edges: int,
    dx_offset: int,
    du_offset: int,
):
    """
    Build rotation generators acting on edge features:

      e = [Δx(3), ||Δx||(1), Δu(3)]

    Rotations act on Δx and Δu, but not on ||Δx||.
    Galilean boosts do not act on edges.

    Parameters
    ----------
    num_edges : int
        Number of edges in the batch.
    dx_offset : int
        Offset of Δx block in flat vector.
    du_offset : int
        Offset of Δu block in flat vector.
    """

    # pure SO(3) generators
    fields, names = generate_euclidean_killing_fields_with_names(
        d=3,
        include_rotations=True,
        include_translations=False,
        backend="torch",
    )

    R_edges = []

    for R in fields:
        # Δx block
        R_dx = lift_field_to_flat_segment(
            R,
            count=num_edges,
            dim=3,
            offset=dx_offset,
        )

        # pad scalar norm slot
        R_dx = pad_field(R_dx, after=1)

        # Δu block
        R_du = lift_field_to_flat_segment(
            R,
            count=num_edges,
            dim=3,
            offset=du_offset,
        )

        # glued representation
        R_edge = sum_fields(R_dx, R_du)
        R_edges.append(R_edge)

    return R_edges



