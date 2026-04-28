# equivariance/__init__.py

from .generators import (
    build_node_generators_velocity,
    build_edge_generators_rotations,
    build_output_generators_velocity,
)

from .packing import (
    pack_flat_nekrs,
    unpack_flat_nekrs,
    compute_flat_offsets,
)

from .model_wrapper import make_model_flat
