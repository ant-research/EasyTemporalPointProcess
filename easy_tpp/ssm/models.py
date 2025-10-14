from typing import Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .initializers import (
    make_DPLR_HiPPO,  # , lecun_normal_ #  init_VinvB, init_log_steps,
)

MATRIX_SCALING_FACTOR = 1


class LLH(nn.Module):
    """
    This is canon:
        L -- number of layers
        N -- number of events.
        P -- Hidden dimension.  Dimensionality of x.
        H -- output dimension.  Dimensionality of y/u.
    """

    def __init__(
        self,
        P: int,
        H: int,
        dt_init_min: float = 1e-4,
        dt_init_max: float = 0.1,  # 1e-3
        dropout_rate: float = 0.0,
        act_func: str = "full_glu",  # F.gelu,
        for_loop: bool = False,
        pre_norm: bool = True,
        post_norm: bool = False,
        simple_mark: bool = True,
        is_first_layer: bool = False,
        relative_time: bool = False,
        complex_values: bool = True,
    ):
        """

        :param P:
        :param H:
        :param dt_init_min:
        :param dt_init_max:
        :param act_func:
        """

        super(LLH, self).__init__()

        # Inscribe the args.
        self.P = P
        self.H = H
        self.dt_init_min = dt_init_min
        self.dt_init_max = dt_init_max
        self.dropout_rate = dropout_rate
        self.complex_values = complex_values

        # select the activation function.
        if act_func == "gelu":
            self.act_func = nn.Sequential(nn.GELU(), nn.Dropout(p=self.dropout_rate))
        elif act_func == "full_glu":
            self.act_func = nn.Sequential(
                nn.Linear(self.H, 2 * self.H),
                nn.Dropout(p=self.dropout_rate),
                nn.GLU(),
                nn.Dropout(p=self.dropout_rate),
            )

        elif (
            act_func == "half_glu"
        ):  # ref: https://github.com/lindermanlab/S5/blob/main/s5/layers.py#L76
            self.act_func1 = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.GELU(),
                nn.Linear(self.H, self.H),
            )
            self.act_func = lambda x: nn.Dropout(p=self.dropout_rate)(
                x * nn.Sigmoid()(self.act_func1(x))
            )
        else:
            raise NotImplementedError(
                "Unrecognized activation function {}".format(act_func)
            )

        # Assume we always use conjugate symmetry.
        self.conj_sym = True

        # Allow a learnable initial state.
        # Needs to be =/= 0 since we take the log to compute
        if self.complex_values:
            self.initial_state_P = nn.Parameter(
                th.complex(
                    th.randn(
                        self.P,
                    ),
                    th.randn(
                        self.P,
                    ),
                )
                * 1e-3,
                requires_grad=True,
            )
        else:
            self.initial_state_P = nn.Parameter(
                th.randn(
                    self.P,
                ),
                requires_grad=True,
            )

        self.norm = nn.LayerNorm(self.H)
        self.for_loop = for_loop
        self.pre_norm = pre_norm
        self.post_norm = post_norm

        self.is_first_layer = is_first_layer
        self.relative_time = relative_time

        self._init_ssm_params()

        self.simple_mark = simple_mark
        if not simple_mark:
            self.mark_a_net = nn.Linear(self.H, self.P, bias=True)
            self.mark_u_net = nn.Linear(
                self.H, self.P, bias=False
            )  # Only need one bias
            self.mark_a_net.weight.data = th.complex(
                nn.init.xavier_normal_(self.mark_a_net.weight.data) * 1e-3,
                nn.init.xavier_normal_(self.mark_a_net.weight.data) * 1e-3,
            )
            self.mark_a_net.bias.data = th.complex(
                nn.init.xavier_normal_(self.mark_a_net.bias.data) * 1e-3,
                nn.init.xavier_normal_(self.mark_a_net.bias.data) * 1e-3,
            )
            self.mark_u_net.weight.data = th.complex(
                nn.init.xavier_normal_(self.mark_u_net.weight.data) * 1e-3,
                nn.init.xavier_normal_(self.mark_u_net.weight.data) * 1e-3,
            )
            if not self.complex_values:
                self.mark_a_net.weight.data = self.mark_a_net.weight.data.real
                self.mark_a_net.bias.data = self.mark_a_net.bias.data.real
                self.mark_u_net.weight.data = self.self.mark_u_net.weight.data.real

    def _init_ssm_params(self):
        self._init_A()
        if not self.is_first_layer:
            self._init_B()
        self._init_C()
        if (
            not self.is_first_layer
        ):  # Could group, but left in same order to not mess with initialization
            self._init_D()
        self._init_E()

    def _init_A(self):
        # Define the initial diagonal HiPPO matrix.
        # Te throw the HiPPO B away.
        Lambda_P, _, _, V_PP, _ = make_DPLR_HiPPO(self.P)
        self.Lambda_P_log_neg_real = th.nn.Parameter((-Lambda_P.real).log())
        self.Lambda_P_imag = th.nn.Parameter(Lambda_P.imag)

        # Store these for use later.
        self._V_PP = V_PP
        self._Vc_PP = V_PP.conj().T

        # We also initialize the step size.
        if self.relative_time:
            self.delta_net = nn.Linear(
                self.H, self.P, bias=True
            )  # nn.Parameter(init_log_steps(self.P, self.dt_init_min, self.dt_init_max))
            with th.no_grad():
                self.delta_net.weight.copy_(
                    nn.init.xavier_normal_(self.delta_net.weight)
                )
                bias = th.ones(
                    self.P,
                )
                bias += th.log(-th.expm1(-bias))
                self.delta_net.bias.copy_(bias)
        else:
            self.log_step_size_P = nn.Parameter(
                th.zeros(size=(self.P,)), requires_grad=False
            )

    @property
    def Lambda_P(self):
        if self.complex_values:
            return th.complex(
                -self.Lambda_P_log_neg_real.exp(),
                self.Lambda_P_imag,
            )
        else:
            return -self.Lambda_P_log_neg_real.exp()

    def _init_B(self):
        # Initialize the B outside the eigenbasis and then transform.
        B = nn.init.xavier_normal_(th.zeros((self.P, self.H))) * MATRIX_SCALING_FACTOR
        B_tilde_PH = self._Vc_PP @ B.type(th.complex64)
        self.B_tilde_PH = (
            th.nn.Parameter(B_tilde_PH)
            if self.complex_values
            else th.nn.Parameter(B_tilde_PH.real)
        )

    def _init_C(self):
        # Use the "complex_normal" initialization.
        # See ~https://github.com/lindermanlab/S5/blob/52cc7e22d6963459ad99a8674e4d3cfb0a480008/s5/ssm.py#L183
        C = nn.init.xavier_normal_(th.zeros((self.H, self.P))) * MATRIX_SCALING_FACTOR
        C_tilde_HP = C.type(th.complex64) @ self._V_PP
        self.C_tilde_HP = (
            th.nn.Parameter(C_tilde_HP)
            if self.complex_values
            else th.nn.Parameter(C_tilde_HP.real)
        )
        # self.C_tilde_HP.data *= 1e-3

    def _init_D(self):
        # Initialize feedthrough (D) matrix. Note the intensity depends on all layers.
        D_HH = th.zeros(self.H)
        nn.init.normal_(D_HH, std=1.0)
        self.D_HH = nn.Parameter(D_HH, requires_grad=True)

    def _init_E(self):
        E = (
            th.nn.init.xavier_normal_(th.zeros((self.P, self.H)))
            * MATRIX_SCALING_FACTOR
        )
        E_tilde_PH = self._Vc_PP @ E.type(th.complex64)
        self.E_tilde_PH = (
            th.nn.Parameter(E_tilde_PH)
            if self.complex_values
            else th.nn.Parameter(E_tilde_PH.real)
        )

    def compute_impulse(self, right_u_H, mark_embedding_H):
        # Compute impulse to add to left limit of x to make right limit.
        alpha_P = th.einsum(
            "ph,...h->...p",
            self.E_tilde_PH,
            mark_embedding_H.type(th.complex64)
            if self.complex_values
            else mark_embedding_H,
        )
        return alpha_P

    def get_lambda(self, right_u_NH, shift_u=True):
        if self.relative_time and (right_u_NH is not None):
            if shift_u:  # during "forward" when dts = [0, t1-t0, ..., t_N-t_{N-1}]
                right_u_NH = F.pad(
                    right_u_NH[..., :-1, :], (0, 0, 1, 0)
                )  # pad default 0 at beginning of second to last dim
            lambda_rescaled_NP = (
                F.softplus(self.delta_net(right_u_NH)) * self.Lambda_P
            )  # predict delta_i from right_u_i
            return {"lambda_rescaled_NP": lambda_rescaled_NP}
        else:
            if self.relative_time:
                lambda_rescaled_P = F.softplus(self.delta_net.bias) * self.Lambda_P
            else:
                lambda_rescaled_P = th.exp(self.log_step_size_P) * self.Lambda_P
            return {"lambda_rescaled_P": lambda_rescaled_P}

    def forward(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        mark_embedding_NH: th.Tensor,
        dt_N: th.Tensor,
        initial_state_P: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Apply the linear SSM to the inputs.

        In the context of TPPs, this returns the right limit of the "intensity function".
        This intensity will have been passed through a non-linearity, though, and so there is no
        guarantee for it is positive.

        :param u_NH: [..., seq_len, input_dim]
        :param alpha_NP: [..., seq_len, hidden_dim]
        :param dt_N: [..., seq_len]
        :param initial_state_P: [..., hidden_dim]
        :return:
        """
        # Pull out the dimensions.
        *leading_dims, _, _ = mark_embedding_NH.shape
        num_leading_dims = len(leading_dims)

        if initial_state_P is None:
            # Pad and expand to match leading dimensions of input
            initial_state_P = self.initial_state_P.view(
                *[1 for _ in range(num_leading_dims)], -1
            ).expand(*leading_dims, -1)

        # Add layer norm
        prime_left_u_NH = left_u_NH
        prime_right_u_NH = right_u_NH
        if prime_left_u_NH is not None:  # ONLY for backward variant
            # assert u_NH.shape[:-1] == alpha_NP[:-1]  # For some reason even when returning True would fail assertion
            assert all(
                u_d == a_d
                for u_d, a_d in zip(prime_left_u_NH.shape, mark_embedding_NH.shape)
            )  # All but last dimensions should match
            if self.pre_norm:
                prime_left_u_NH = self.norm(prime_left_u_NH)
        if prime_right_u_NH is not None:
            # assert u_NH.shape[:-1] == alpha_NP[:-1]  # For some reason even when returning True would fail assertion
            assert all(
                u_d == a_d
                for u_d, a_d in zip(prime_right_u_NH.shape, mark_embedding_NH.shape)
            )  # All but last dimensions should match
            if self.pre_norm:
                prime_right_u_NH = self.norm(prime_right_u_NH)

        right_x_NP, left_y_NH, right_y_NH = self._ssm(
            left_u_NH=prime_left_u_NH,
            right_u_NH=prime_right_u_NH,
            impulse_NP=self.compute_impulse(prime_right_u_NH, mark_embedding_NH),
            dt_N=dt_N,
            initial_state_P=initial_state_P,
        )
        # Given the following:
        # right_u: u0, u1, u2, ... <-> u_{t_0}, u_{t_1}, u_{t_2}, ...
        # left_u: u0, u1, u2, ... <-> u_{t_0-}, u_{t_1-}, u_{t_2-}, ...
        # a: a0, a1, a2, ... <-> mark embeddings for m_0, m_1, m_2, ... at times t_0, t_1, t_2
        # dt: dt0, dt1, dt2, ... <-> 0, t_1-t_0, t_2-t_1, ...
        # initial_state_p: hidden state to evolve to to compute x_{0}
        # Returns the following:
        # right_x: x0, x1, x2, ... <-> x_{t_0}, x_{t_1}, x_{t_2}, ...
        # right_y: y0, y1, y2, ... <-> y_{t_0}, y_{t_1}, y_{t_2}, ...
        # left_y: y0, y1, y2, ... <-> y_{t_0-}, y_{t_1-}, y_{t_2-}, ...

        next_layer_left_u_NH = next_layer_right_u_NH = None
        if left_y_NH is not None:
            next_layer_left_u_NH = self.act_func(left_y_NH) + (
                left_u_NH if left_u_NH is not None else 0.0
            )
            if self.post_norm:
                next_layer_left_u_NH = self.norm(next_layer_left_u_NH)
        if right_y_NH is not None:
            next_layer_right_u_NH = self.act_func(right_y_NH) + (
                right_u_NH if right_u_NH is not None else 0.0
            )
            if self.post_norm:
                next_layer_right_u_NH = self.norm(next_layer_right_u_NH)
        return right_x_NP, next_layer_left_u_NH, next_layer_right_u_NH

    def _ssm(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        impulse_NP: th.Tensor,
        dt_N: th.Tensor,  # [0, t_1 - t_0, ..., t_N - t_{N-1}]
        initial_state_P: th.Tensor,
    ):
        *leading_dims, N, P = impulse_NP.shape
        u_NH = right_u_NH  # This implementation does not use left_u, nor does it compute left_y
        if u_NH is not None:
            impulse_NP = impulse_NP + th.einsum(
                "ph,...nh->...np",
                self.B_tilde_PH,
                u_NH.type(th.complex64) if self.complex_values else u_NH,
            )
            y_u_res_NH = th.einsum(
                "...nh,h->...nh", u_NH, self.D_HH
            )  # D_HH should really be D_H
        else:
            assert self.is_first_layer
            y_u_res_NH = 0.0

        lambda_res = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        if "lambda_rescaled_P" in lambda_res:  # original formulation
            lambda_dt_NP = th.einsum(
                "...n,p->...np", dt_N, lambda_res["lambda_rescaled_P"]
            )
        else:  # relative time
            lambda_dt_NP = th.einsum(
                "...n,...np->...np", dt_N, lambda_res["lambda_rescaled_NP"]
            )

        if self.for_loop:
            right_x_P = initial_state_P
            right_x_NP = []
            for i in range(N):
                right_x_P = (
                    lambda_dt_NP[..., i, :].exp() * right_x_P + impulse_NP[..., i, :]
                )
                right_x_NP.append(right_x_P)
            right_x_NP = th.stack(right_x_NP, dim=-2)
        else:
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), impulse_NP), dim=-2
            ).log()
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0))
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star
            )
            right_x_NP = right_x_log_NP.exp()[..., 1:, :]

        conj_sym_mult = 2 if self.conj_sym else 1
        y_NH = (
            conj_sym_mult
            * th.einsum("...np,hp->...nh", right_x_NP, self.C_tilde_HP).real
            + y_u_res_NH
        )

        return right_x_NP, None, y_NH

    def get_left_limit(
        self,
        right_limit_P: th.Tensor,  # Along with dt, can have any number of leading dimensions, produces a tensor of dim ...MP
        dt_G: th.Tensor,
        current_right_u_H: th.Tensor,
        next_left_u_GH: th.Tensor,
    ) -> th.Tensor:
        """
        To get the left limit, we roll on the layer for the right dt.
        Computed for a single point (vmap for multiple).

        TODO - its kind of messy that this repeats from forward.
         Find a way to re-use that code.

        :param right_limit_P: at [t_0, ..., t_{N-1}]
        :param dt: Length of time to roll the layer on for. at [t_1 - t_0, ..., t_N - t_{N-1}]
        :param current_right_u_H: at [t_0, ..., t_{N-1}] -- for relative-time variant
        :param next_left_u_GH: at [t_1, ..., t_N] -- for backward variant

        :return:
        """

        if current_right_u_H is not None and self.pre_norm:
            current_right_u_H = self.norm(current_right_u_H)

        lambda_res = self.get_lambda(
            current_right_u_H, shift_u=False
        )  # U should already be shifted
        if "lambda_rescaled_P" in lambda_res:
            lambda_bar_GP = th.exp(
                th.einsum("...g,p->...gp", dt_G, lambda_res["lambda_rescaled_P"])
            )
        else:
            lambda_bar_GP = th.exp(
                th.einsum("...g,...p->...gp", dt_G, lambda_res["lambda_rescaled_NP"])
            )

        return th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP)

    def depth_pass(
        self,
        current_left_x_P: th.Tensor,  # No leading dimensions (seq, batch, etc.) here because we accommodate any of them
        current_left_u_H: Optional[
            th.Tensor
        ],  # Just assume that x and u match in the leading dimensions. Produces y_H with equivalent leading dimensions
        prev_right_u_H: Optional[
            th.Tensor
        ],  # Just assume that x and u match in the leading dimensions. Produces y_H with equivalent leading dimensions
    ) -> th.Tensor:
        if current_left_u_H is not None:
            if self.pre_norm:
                prime_u_H = self.norm(current_left_u_H)
            else:
                prime_u_H = current_left_u_H
            y_u_res_H = th.einsum(
                "...h,h->...h", prime_u_H, self.D_HH
            )  # D_HH should really be D_H
        else:
            assert self.is_first_layer
            y_u_res_H = 0.0

        conj_sym_mult = 2 if self.conj_sym else 1
        y_H = (
            conj_sym_mult
            * th.einsum("...p,hp->...h", current_left_x_P, self.C_tilde_HP).real
            + y_u_res_H
        )

        # Apply an activation function.
        if self.post_norm:
            new_u_H = self.norm(
                self.act_func(y_H)
                + (current_left_u_H if current_left_u_H is not None else 0.0)
            )
        else:
            new_u_H = self.act_func(y_H) + (
                current_left_u_H if current_left_u_H is not None else 0.0
            )

        return new_u_H


class Int_Forward_LLH(LLH):
    # LLH but Bu_t is integrated w.r.t dt instead of dN_t
    # After discretization, when evolving x_t to x_t', applies ZOH on u_t over [t,t'] forward in time
    # (as opposed to u_{t'} backwards over [t,t'])

    def _ssm(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        impulse_NP: th.Tensor,
        dt_N: th.Tensor,
        initial_state_P: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Apply the linear SSM to the inputs.

        In the context of TPPs, this returns the right limit of the "intensity function".
        This intensity will have been passed through a non-linearity, though, and so there is no
        guarantee for it is positive.

        :param u_NH: [..., seq_len, input_dim]
        :param alpha_NP: [..., seq_len, hidden_dim]
        :param dt_N: [..., seq_len]
        :param initial_state_P: [..., hidden_dim]

        :return:
        """
        # Pull out the dimensions.
        *leading_dims, N, P = impulse_NP.shape

        lambda_res = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        if "lambda_rescaled_P" in lambda_res:
            lambda_rescaled = lambda_res["lambda_rescaled_P"]
            lambda_dt_NP = th.einsum(
                "...n,p->...np", dt_N, lambda_res["lambda_rescaled_P"]
            )
        else:
            lambda_rescaled = lambda_res["lambda_rescaled_NP"]
            lambda_dt_NP = th.einsum(
                "...n,...np->...np", dt_N, lambda_res["lambda_rescaled_NP"]
            )

        if left_u_NH is not None:
            left_Du_NH = th.einsum(
                "...nh,h->...nh",
                left_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            left_Du_NH = 0.0

        if right_u_NH is not None:
            right_u_NH = F.pad(right_u_NH[..., :-1, :], (0, 0, 1, 0))
            right_Bu_NP = th.einsum(
                "...np,ph,...nh->...np",
                lambda_dt_NP.exp() - 1.0,  # dts: [0, t1-t0, t2-t1, ...]
                self.B_tilde_PH,
                right_u_NH.type(th.complex64) if self.complex_values else right_u_NH,
            )
            right_Du_NH = th.einsum(
                "...nh,h->...nh",
                right_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            right_Bu_NP = right_Du_NH = 0.0

        if self.for_loop:
            right_x_P = initial_state_P
            left_x_NP, right_x_NP = [], []
            for i in range(N):
                # TODO: check left_u_NH starts at t_0 or t_1
                left_x_P = lambda_dt_NP[..., i, :].exp() * right_x_P + (
                    right_Bu_NP[..., i, :] if left_u_NH is not None else 0.0
                )
                right_x_P = left_x_P + impulse_NP[..., i, :]
                left_x_NP.append(left_x_P)
                right_x_NP.append(right_x_P)
            right_x_NP = th.stack(
                right_x_NP, dim=-2
            )  # discard initial_hidden_states, right_limit of xs for [t0, t1, ...]
            left_x_NP = th.stack(
                left_x_NP, dim=-2
            )  # discard initial_hidden_states, left_limit of xs for [t0, t1, ...]
        else:
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), right_Bu_NP + impulse_NP), dim=-2
            ).log()
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0))
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star
            )
            right_x_NP = right_x_log_NP.exp()  # Contains initial_state_P in index 0
            left_x_NP = (
                lambda_dt_NP.exp() * right_x_NP[..., :-1, :] + right_Bu_NP
            )  # Evolves previous hidden state forward to compute left limit
            right_x_NP = right_x_NP[..., 1:, :]

        conj_sym_mult = 2 if self.conj_sym else 1
        left_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, left_x_NP).real
            + left_Du_NH
        )  # ys for [t0, t1, ...]
        right_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, right_x_NP).real
            + right_Du_NH
        )  # ys for [t0, t1, ...]

        return right_x_NP, left_y_NH, right_y_NH

    def get_left_limit(
        self,
        right_limit_P: th.Tensor,  # Along with dt, can have any number of leading dimensions, produces a tensor of dim ...MP
        dt_G: th.Tensor,
        current_right_u_H: Optional[th.Tensor],
        next_left_u_GH: Optional[th.Tensor],
    ) -> th.Tensor:
        """
        To get the left limit, we roll on the layer for the right dt.
        Computed for a single point (vmap for multiple).

        TODO - its kind of messy that this repeats from forward.
         Find a way to re-use that code.

        :param right_limit_P:
        :param dt: Length of time to roll the layer on for.
        :return:
        """
        if current_right_u_H is not None and self.pre_norm:
            current_right_u_H = self.norm(current_right_u_H)

        lambda_res = self.get_lambda(
            current_right_u_H, shift_u=False
        )  # U should already be shifted
        if "lambda_rescaled_P" in lambda_res:
            lambda_rescaled = lambda_res["lambda_rescaled_P"]
            lambda_bar_GP = th.exp(
                th.einsum("...g,p->...gp", dt_G, lambda_res["lambda_rescaled_P"])
            )
        else:
            lambda_rescaled = lambda_res["lambda_rescaled_NP"]
            lambda_bar_GP = th.exp(
                th.einsum("...g,...p->...gp", dt_G, lambda_res["lambda_rescaled_NP"])
            )

        # lambda_rescaled_P = th.exp(self.log_step_size_P) * self.Lambda_P
        # lambda_bar_GP = th.exp(th.einsum('...g,p->...gp', dt_G, lambda_rescaled_P))
        int_hidden_GP = th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP)

        if current_right_u_H is None:  # no Bu term
            assert self.is_first_layer
            return int_hidden_GP
        else:  # add Bu to impulse
            if self.pre_norm:
                current_right_u_H = self.norm(current_right_u_H)

            impulse_GP = th.einsum(
                "...gp,ph,...h->...gp",
                lambda_bar_GP - 1.0,
                self.B_tilde_PH,
                current_right_u_H.type(th.complex64)
                if self.complex_values
                else current_right_u_H,
            )

            return int_hidden_GP + impulse_GP


class Int_Backward_LLH(Int_Forward_LLH):
    # LLH but Bu_t is integrated w.r.t dt instead of dN_t
    # After discretization, when evolving x_t to x_t', applies ZOH on u_t' over [t,t'] backwards in time
    # (as opposed to u_{t} forwards over [t,t'])

    def _ssm(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        impulse_NP: th.Tensor,
        dt_N: th.Tensor,
        initial_state_P: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Apply the linear SSM to the inputs.

        In the context of TPPs, this returns the right limit of the "intensity function".
        This intensity will have been passed through a non-linearity, though, and so there is no
        guarantee for it is positive.

        :param u_NH: [..., seq_len, input_dim]
        :param alpha_NP: [..., seq_len, hidden_dim]
        :param dt_N: [..., seq_len]
        :param initial_state_P: [..., hidden_dim]

        :return:
        """
        # Pull out the dimensions.
        *leading_dims, N, P = impulse_NP.shape

        # lambda_rescaled_P = th.exp(self.log_step_size_P) * self.Lambda_P
        # lambda_dt_NP = th.einsum('...n,p->...np', dt_N, lambda_rescaled_P)
        lambda_res = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        if "lambda_rescaled_P" in lambda_res:
            lambda_rescaled = lambda_res["lambda_rescaled_P"]
            lambda_dt_NP = th.einsum(
                "...n,p->...np", dt_N, lambda_res["lambda_rescaled_P"]
            )
        else:
            lambda_rescaled = lambda_res["lambda_rescaled_NP"]
            lambda_dt_NP = th.einsum(
                "...n,...np->...np", dt_N, lambda_res["lambda_rescaled_NP"]
            )

        if left_u_NH is not None:
            left_Bu_NP = th.einsum(  # try a different mapping
                "...np,ph,...nh->...np",
                lambda_dt_NP.exp() - 1.0,  # dts: [0, t1-t0, t2-t1, ...]
                self.B_tilde_PH,
                left_u_NH.type(th.complex64) if self.complex_values else left_u_NH,
            )
            left_Du_NH = th.einsum(
                "...nh,h->...nh",
                left_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            left_Bu_NP = left_Du_NH = 0.0

        if right_u_NH is not None:
            right_Du_NH = th.einsum(
                "...nh,h->...nh",
                right_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            right_Du_NH = 0.0

        if self.for_loop:
            right_x_P = initial_state_P
            left_x_NP, right_x_NP = [], []
            for i in range(N):
                left_x_P = lambda_dt_NP[..., i, :].exp() * right_x_P + (
                    left_Bu_NP[..., i, :] if left_u_NH is not None else 0.0
                )
                right_x_P = left_x_P + impulse_NP[..., i, :]
                left_x_NP.append(left_x_P)
                right_x_NP.append(right_x_P)
            right_x_NP = th.stack(
                right_x_NP, dim=-2
            )  # discard initial_hidden_states, right_limit of xs for [t0, t1, ...]
            left_x_NP = th.stack(
                left_x_NP, dim=-2
            )  # discard initial_hidden_states, left_limit of xs for [t0, t1, ...]
        else:
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), left_Bu_NP + impulse_NP), dim=-2
            ).log()
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0))
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star
            )
            right_x_NP = right_x_log_NP.exp()  # Contains initial_state_P in index 0
            left_x_NP = (
                lambda_dt_NP.exp() * right_x_NP[..., :-1, :] + left_Bu_NP
            )  # Evolves previous hidden state forward to compute left limit
            right_x_NP = right_x_NP[..., 1:, :]

        conj_sym_mult = 2 if self.conj_sym else 1
        left_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, left_x_NP).real
            + left_Du_NH
        )  # ys for [t0, t1, ...]
        right_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, right_x_NP).real
            + right_Du_NH
        )  # ys for [t0, t1, ...]

        return right_x_NP, left_y_NH, right_y_NH

    def get_left_limit(
        self,
        right_limit_P: th.Tensor,  # Along with dt, can have any number of leading dimensions, produces a tensor of dim ...MP
        dt_G: th.Tensor,
        current_right_u_H: th.Tensor,
        next_left_u_GH: th.Tensor,
    ) -> th.Tensor:
        """
        To get the left limit, we roll on the layer for the right dt.
        Computed for a single point (vmap for multiple).

        TODO - its kind of messy that this repeats from forward.
         Find a way to re-use that code.

        :param right_limit_P:
        :param dt: Length of time to roll the layer on for.
        :return:
        """

        if current_right_u_H is not None and self.pre_norm:
            current_right_u_H = self.norm(current_right_u_H)

        lambda_res = self.get_lambda(
            current_right_u_H, shift_u=False
        )  # U should already be shifted
        if "lambda_rescaled_P" in lambda_res:
            lambda_rescaled = lambda_res["lambda_rescaled_P"]
            lambda_bar_GP = th.exp(
                th.einsum("...g,p->...gp", dt_G, lambda_res["lambda_rescaled_P"])
            )
        else:
            lambda_rescaled = lambda_res["lambda_rescaled_NP"]
            lambda_bar_GP = th.exp(
                th.einsum("...g,...p->...gp", dt_G, lambda_res["lambda_rescaled_NP"])
            )

        int_hidden_GP = th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP)

        if next_left_u_GH is None:  # no Bu term
            assert self.is_first_layer
            return int_hidden_GP
        else:  # add Bu to impulse
            if self.pre_norm:
                next_left_u_GH = self.norm(next_left_u_GH)

            impulse_GP = th.einsum(  # try a different mapping
                "...gp,ph,...gh->...gp",
                lambda_bar_GP - 1.0,
                self.B_tilde_PH,
                next_left_u_GH.type(th.complex64)
                if self.complex_values
                else next_left_u_GH,
            )

            return int_hidden_GP + impulse_GP
