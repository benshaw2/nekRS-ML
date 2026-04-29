# equivariance/generators.py

import math
from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.enforcement.regularization import (
    as_field_lastdim,
    lift_field_to_flat_segment,
    sum_fields,
)

import torch

def build_node_generators_velocity(with_names=False):
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

    X = [as_field_lastdim(f, d=3) for f in fields]

    # act on last dimension
    #X_nodes = [as_field_lastdim(f, d=3) for f in fields]
    if with_names:
        return X, names
    return X



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


def build_output_generators_mandel_stress():
    labels = ["w1","w2","w3","w4","w5","w6"]
    pair = _pairer_from_labels(labels)
    d = 6

    fields, names = generate_euclidean_killing_fields_with_names(
        d=d,
        include_translations=False,
        include_rotations=True,
        backend="torch"
    )

    name_to_field = {n: f for f, n in zip(fields, names)}

    def R(a, b):
        i, j = pair(a, b)
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return as_field_lastdim(name_to_field[key], d=d)

    s2 = math.sqrt(2.0)

    Y1 = sum_fields(
        R("w2","w4"),   # (T12, T22)
        R("w1","w2"),   # (T11, T12)
        R("w3","w5"),   # (T13, T23)
        weights=[s2, s2, 1.0]
    )

    Y2 = sum_fields(
        R("w1","w3"),
        R("w3","w6"),
        R("w2","w5"),
        weights=[s2, s2, 1.0]
    )

    Y3 = sum_fields(
        R("w4","w5"),
        R("w5","w6"),
        R("w2","w3"),
        weights=[s2, s2, 1.0]
    )

    return Y1, Y2, Y3


def pad_field(field, *, before=0, after=0):
    """
    Pad a vector field with invariant (zero) dimensions
    before and/or after its action.

    Parameters
    ----------
    field : callable
        A vector field acting on ℝ^d
    before : int
        Number of zero dimensions prepended
    after : int
        Number of zero dimensions appended
    """

    if before == 0 and after == 0:
        return field

    def padded(x, *, meta=None, grad=None):
        '''# x is the *full* flat vector
        # extract the slice this field acts on
        core = x[before : x.shape[0] - after]
        delta_core = field(core)

        # assemble full delta
        delta = torch.zeros_like(x)
        delta[before : before + delta_core.shape[0]] = delta_core
        return delta'''

        # x: (B, D)
        B, D = x.shape

        # slice out the active block
        core = x[:, before : D - after]

        delta_core = field(core, meta=meta, grad=grad)

        delta = torch.zeros_like(x)
        delta[:, before : before + delta_core.shape[1]] = delta_core
        return delta


    return padded
