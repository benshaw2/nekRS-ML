import torch
import math

_SQRT2 = math.sqrt(2.0)

def mat_to_mandel(T):
    """
    Convert symmetric (..., 3, 3) tensor to Mandel (..., 6).

    Ordering:
      [T11, sqrt(2)*T12, sqrt(2)*T13, T22, sqrt(2)*T23, T33]
    """
    return torch.stack(
        [
            T[..., 0, 0],              # T11
            _SQRT2 * T[..., 0, 1],     # sqrt(2) * T12
            _SQRT2 * T[..., 0, 2],     # sqrt(2) * T13
            T[..., 1, 1],              # T22
            _SQRT2 * T[..., 1, 2],     # sqrt(2) * T23
            T[..., 2, 2],              # T33
        ],
        dim=-1,
    )


def mandel_to_mat(w):
    """
    Convert Mandel (..., 6) vector to symmetric (..., 3, 3) tensor.
    Inverse of mat_to_mandel.
    """
    T = torch.zeros(
        (*w.shape[:-1], 3, 3),
        dtype=w.dtype,
        device=w.device,
    )

    T[..., 0, 0] = w[..., 0]                 # T11
    T[..., 0, 1] = w[..., 1] / _SQRT2        # T12
    T[..., 1, 0] = w[..., 1] / _SQRT2

    T[..., 0, 2] = w[..., 2] / _SQRT2        # T13
    T[..., 2, 0] = w[..., 2] / _SQRT2

    T[..., 1, 1] = w[..., 3]                 # T22

    T[..., 1, 2] = w[..., 4] / _SQRT2        # T23
    T[..., 2, 1] = w[..., 4] / _SQRT2

    T[..., 2, 2] = w[..., 5]                 # T33

    return T


def mandel_log(T):
    return mat_to_mandel(torch.linalg.matrix_log(T))

def mandel_exp(m):
    T = mandel_to_mat(m)
    return torch.linalg.matrix_exp(T)