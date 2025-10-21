# @title Imports and environment
import torch as th


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.

    modified from: https://github.com/lindermanlab/S5/blob/3c18fdb6b06414da35e77b94b9cd855f6a95ef17/s5/ssm.py#L29

    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = th.ones(Lambda.shape[0])
    Lambda_bar = th.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def apply_ssm(
    Lambda_bar_NP,
    B_bar_NPH,
    C_tilde_HP,
    input_sequence_NH,
    alpha_NP,
    conj_sym,
    initial_state_P=None,
):
    """Compute the NxH output of discretized SSM given an NxH input.

    modified from: https://github.com/lindermanlab/S5/blob/3c18fdb6b06414da35e77b94b9cd855f6a95ef17/s5/ssm.py#L60
    - removed bidirectionality.
    - assume Lambda_bar is N-length.

    Args:
        Lambda_bar_NP (complex64):      discretized diagonal state matrix for each interval     (N, P)
        B_bar_NPH (complex64):          "discretized" input matrix.  Note: may be outside ZOH   (N, P, H)
        C_tilde_HP (complex64):         output matrix                                           (H, P)
        input_sequence_NH (float32):    input sequence of features                              (N, H)
        alpha_NP (complex64):           mark-specific biases                                    (N, P)
        conj_sym (bool):                Whether conjugate symmetry is enforced
        initial_state_P ():             Allow passing in a specific initial state (otherwise zero is assumed.)
    Returns:
        ys_NH (float32): the SSM outputs (S5 layer preactivations)      (N, H)
    """
    N, P, H = B_bar_NPH.shape

    # Compute effective inputs.
    Bu_elements_NP = th.vmap(lambda b, u, alpha: b @ u.type(th.complex64) + alpha)(
        B_bar_NPH, input_sequence_NH, alpha_NP
    )

    # # Torch doesn't roll an associative scan... yet...
    # _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    # Set the initial state if we haven't already.
    if initial_state_P is None:
        state = th.zeros((P,))
    else:
        state = initial_state_P

    # Accumulate the hidden states here.  Note the initial state shouldn't be returned.
    # xs = th.zeros((L, P)).type(th.complex64)
    xs = [state]

    for i, (lam_P, bu_P) in enumerate(zip(Lambda_bar_NP, Bu_elements_NP)):
        # state = lam_P * state + bu_P
        # xs[i] = state
        xs.append(lam_P * xs[-1] + bu_P)
    xs = th.stack(xs)[1:]

    # Output the xs and ys after projecting.
    if conj_sym:
        return xs, th.vmap(lambda x: 2 * (C_tilde_HP @ x).real)(xs)
    else:
        return xs, th.vmap(lambda x: (C_tilde_HP @ x).real)(xs)
