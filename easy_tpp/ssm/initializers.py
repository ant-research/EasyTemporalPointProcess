import math

import numpy as np
import numpy as onp
import torch as th
from numpy.linalg import eigh


def make_HiPPO(P):
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        P (int32): state size
    Returns:
        P x P HiPPO LegS matrix
    """
    M = np.sqrt(1 + 2 * np.arange(P))
    A = M[:, np.newaxis] * M[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(P))
    return -A


def make_NPLR_HiPPO(P):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        P (int32): state size

    Returns:
        P x P HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(P)

    # Add in a rank 1 term. Makes it Normal.
    R1 = np.sqrt(np.arange(P) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(P) + 1.0)
    return hippo, R1, B


def make_DPLR_HiPPO(P):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        P:

    Returns:
        eigenvalues Lambda, low-rank term R1, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, R1, B = make_NPLR_HiPPO(P)

    S = A + R1[:, np.newaxis] * R1[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    R1 = V.conj().T @ R1
    B_orig = B
    B = V.conj().T @ B
    return (
        th.tensor(onp.asarray(Lambda_real + 1j * Lambda_imag), dtype=th.complex64),
        th.tensor(onp.asarray(R1)),
        th.tensor(onp.asarray(B)),
        th.tensor(onp.asarray(V), dtype=th.complex64),
        th.tensor(onp.asarray(B_orig)),
    )


def init_log_steps(P, dt_min, dt_max):
    """Initialize an array of learnable timescale parameters.
    initialized uniformly in log space.
     Args:
         input:
     Returns:
         initialized array of timescales (float32): (P,)
    """
    unlog = th.rand(size=(P,))
    log = unlog * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    return log


def lecun_normal_(tensor: th.Tensor) -> th.Tensor:
    input_size = tensor.shape[
        -1
    ]  # Assuming that the weights' input dimension is the last.
    std = math.sqrt(1 / input_size)
    with th.no_grad():
        return tensor.normal_(0, std)  # or torch.nn.init.xavier_normal_


def init_VinvB(shape, Vinv):
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex

    Modified from https://github.com/lindermanlab/S5/blob/52cc7e22d6963459ad99a8674e4d3cfb0a480008/s5/ssm.py#L165

    numbers.
     Args:
         shape (tuple): desired shape  (P,H)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (P,H)
    """
    B = lecun_normal_(th.zeros(shape))
    VinvB = Vinv @ B.type(th.complex64)
    return VinvB
