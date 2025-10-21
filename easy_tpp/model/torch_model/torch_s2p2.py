from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.ssm.models import LLH, Int_Backward_LLH, Int_Forward_LLH


class ComplexEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ComplexEmbedding, self).__init__()
        self.real_embedding = nn.Embedding(*args, **kwargs)
        self.imag_embedding = nn.Embedding(*args, **kwargs)

        self.real_embedding.weight.data *= 1e-3
        self.imag_embedding.weight.data *= 1e-3

    def forward(self, x):
        return torch.complex(
            self.real_embedding(x),
            self.imag_embedding(x),
        )


class IntensityNet(nn.Module):
    def __init__(self, input_dim, bias, num_event_types):
        super().__init__()
        self.intensity_net = nn.Linear(input_dim, num_event_types, bias=bias)
        self.softplus = ScaledSoftplus(num_event_types)

    def forward(self, x):
        return self.softplus(self.intensity_net(x))


class S2P2(TorchBaseModel):
    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(S2P2, self).__init__(model_config)
        self.n_layers = model_config.num_layers
        self.P = model_config.model_specs["P"]  # Hidden state dimension
        self.H = model_config.hidden_size  # Residual stream dimension
        self.beta = model_config.model_specs.get("beta", 1.0)
        self.bias = model_config.model_specs.get("bias", True)
        # self.softplus = torch.nn.Softplus(beta=self.beta)
        self.simple_mark = model_config.model_specs.get("simple_mark", True)

        layer_kwargs = dict(
            P=self.P,
            H=self.H,
            dt_init_min=model_config.model_specs.get("dt_init_min", 1e-4),
            dt_init_max=model_config.model_specs.get("dt_init_max", 0.1),
            act_func=model_config.model_specs.get("act_func", "full_glu"),
            dropout_rate=model_config.model_specs.get("dropout_rate", 0.0),
            for_loop=model_config.model_specs.get("for_loop", False),
            pre_norm=model_config.model_specs.get("pre_norm", True),
            post_norm=model_config.model_specs.get("post_norm", False),
            simple_mark=self.simple_mark,
            relative_time=model_config.model_specs.get("relative_time", False),
            complex_values=model_config.model_specs.get("complex_values", True),
        )

        int_forward_variant = model_config.model_specs.get("int_forward_variant", False)
        int_backward_variant = model_config.model_specs.get(
            "int_backward_variant", False
        )
        assert (
            int_forward_variant + int_backward_variant
        ) <= 1  # Only one at most is allowed to be specified

        if int_forward_variant:
            llh_layer = Int_Forward_LLH
        elif int_backward_variant:
            llh_layer = Int_Backward_LLH
        else:
            llh_layer = LLH

        self.backward_variant = int_backward_variant

        self.layers = nn.ModuleList(
            [
                llh_layer(**layer_kwargs, is_first_layer=i == 0)
                for i in range(self.n_layers)
            ]
        )
        self.layers_mark_emb = nn.Embedding(
            self.num_event_types_pad,
            self.H,
        )  # One embedding to share amongst layers to be used as input into a layer-specific and input-aware impulse
        self.layer_type_emb = None  # Remove old embeddings from EasyTPP
        self.intensity_net = IntensityNet(
            input_dim=self.H,
            bias=self.bias,
            num_event_types=self.num_event_types,
        )

    def _get_intensity(
        self, x_LP: Union[torch.tensor, List[torch.tensor]], right_us_BNH
    ) -> torch.Tensor:
        """
        Assume time has already been evolved, take a vertical stack of hidden states and produce intensity.
        """
        left_u_H = None
        for i, layer in enumerate(self.layers):
            if isinstance(
                x_LP, list
            ):  # Sometimes it is convenient to pass as a list over the layers rather than a single tensor
                left_u_H = layer.depth_pass(
                    x_LP[i], current_left_u_H=left_u_H, prev_right_u_H=right_us_BNH[i]
                )
            else:
                left_u_H = layer.depth_pass(
                    x_LP[..., i, :],
                    current_left_u_H=left_u_H,
                    prev_right_u_H=right_us_BNH[i],
                )

        return self.intensity_net(left_u_H)  # self.ScaledSoftplus(self.linear(left_u_H))

    def _evolve_and_get_intensity_at_sampled_dts(self, x_LP, dt_G, right_us_H):
        left_u_GH = None
        for i, layer in enumerate(self.layers):
            x_GP = layer.get_left_limit(
                right_limit_P=x_LP[..., i, :],
                dt_G=dt_G,
                next_left_u_GH=left_u_GH,
                current_right_u_H=right_us_H[i],
            )
            left_u_GH = layer.depth_pass(
                current_left_x_P=x_GP,
                current_left_u_H=left_u_GH,
                prev_right_u_H=right_us_H[i],
            ) 
        return self.intensity_net(left_u_GH)  # self.ScaledSoftplus(self.linear(left_u_GH))

    def forward(
        self, batch, initial_state_BLP: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch operations of self._forward
        """
        t_BN, dt_BN, marks_BN, batch_non_pad_mask, _ = batch

        right_xs_BNP = []  # including both t_0 and t_N
        left_xs_BNm1P = []
        right_us_BNH = [
            None
        ]  # Start with None as this is the 'input' to the first layer
        left_u_BNH, right_u_BNH = None, None
        alpha_BNP = self.layers_mark_emb(marks_BN)

        for l_i, layer in enumerate(self.layers):
            # for each event, compute the fixed impulse via alpha_m for event i of type m
            init_state = (
                initial_state_BLP[:, l_i] if initial_state_BLP is not None else None
            )

            # Returns right limit of xs and us for [t0, t1, ..., tN]
            # "layer" returns the right limit of xs at current layer, and us for the next layer (as transformations of ys)
            # x_BNP: at time [t_0, t_1, ..., t_{N-1}, t_N]
            # next_left_u_BNH: at time [t_0, t_1, ..., t_{N-1}, t_N] -- only available for backward variant
            # next_right_u_BNH: at time [t_0, t_1, ..., t_{N-1}, t_N] -- always returned but only used for RT
            x_BNP, next_layer_left_u_BNH, next_layer_right_u_BNH = layer.forward(
                left_u_BNH, right_u_BNH, alpha_BNP, dt_BN, init_state
            )
            assert next_layer_right_u_BNH is not None

            right_xs_BNP.append(x_BNP)
            if next_layer_left_u_BNH is None:  # NOT backward variant
                left_xs_BNm1P.append(
                    layer.get_left_limit(  # current and next at event level
                        x_BNP[..., :-1, :],  # at time [t_0, t_1, ..., t_{N-1}]
                        dt_BN[..., 1:].unsqueeze(
                            -1
                        ),  # with dts [t1-t0, t2-t1, ..., t_N-t_{N-1}]
                        current_right_u_H=right_u_BNH
                        if right_u_BNH is None
                        else right_u_BNH[
                            ..., :-1, :
                        ],  # at time [t_0, t_1, ..., t_{N-1}]
                        next_left_u_GH=left_u_BNH
                        if left_u_BNH is None
                        else left_u_BNH[..., 1:, :].unsqueeze(
                            -2
                        ),  # at time [t_1, t_2 ..., t_N]
                    ).squeeze(-2)
                )
            right_us_BNH.append(next_layer_right_u_BNH)

            left_u_BNH, right_u_BNH = next_layer_left_u_BNH, next_layer_right_u_BNH

        right_xs_BNLP = torch.stack(right_xs_BNP, dim=-2)

        ret_val = {
            "right_xs_BNLP": right_xs_BNLP,  # [t_0, ..., t_N]
            "right_us_BNH": right_us_BNH,  # [t_0, ..., t_N]; list starting with None
        }

        if left_u_BNH is not None:  # backward variant
            ret_val["left_u_BNm1H"] = left_u_BNH[
                ..., 1:, :
            ]  # The next inputs after last layer -> transformation of ys
        else:  # NOT backward variant
            ret_val["left_xs_BNm1LP"] = torch.stack(left_xs_BNm1P, dim=-2)

        # 'seq_len - 1' left limit for [t_1, ..., t_N] for events (u if available, x if not)
        # 'seq_len' right limit for [t_0, t_1, ..., t_{N-1}, t_N] for events xs or us
        return ret_val

    def loglike_loss(self, batch, **kwargs):
        # hidden states at the left and right limits around event time; note for the shift by 1 in indices:
        # consider a sequence [t0, t1, ..., tN]
        # Produces the following:
        # left_x: x0, x1, x2, ... <-> x_{t_1-}, x_{t_2-}, x_{t_3-}, ..., x_{t_N-} (note the shift in indices) for all layers
        #    OR ==>               <-> u_{t_1-}, u_{t_2-}, u_{t_3-}, ..., u_{t_N-} for last layer
        #
        # right_x: x0, x1, x2, ... <-> x_{t_0+}, x_{t_1+}, ..., x_{t_N+} for all layers
        # right_u: u0, u1, u2, ... <-> u_{t_0+}, u_{t_1+}, ..., u_{t_N+} for all layers
        forward_results = self.forward(
            batch
        )  # N minus 1 comparing with sequence lengths
        right_xs_BNLP, right_us_BNH = (
            forward_results["right_xs_BNLP"],
            forward_results["right_us_BNH"],
        )
        right_us_BNm1H = [
            None if right_u_BNH is None else right_u_BNH[:, :-1, :]
            for right_u_BNH in right_us_BNH
        ]

        event_times_BN, inter_event_times_BN, marks_BN, batch_non_pad_mask, _ = batch

        # evaluate intensity values at each event *from the left limit*, _get_intensity: [LP] -> [M]
        # left_xs_B_Nm1_LP = left_xs_BNm1LP[:, :-1, ...]  # discard the left limit of t_N
        # Note: no need to discard the left limit of t_N because "marks_mask" will deal with it
        if "left_u_BNm1H" in forward_results:  # ONLY backward variant
            intensity_B_Nm1_M = self.intensity_net(
                forward_results["left_u_BNm1H"]
            )  # self.softplus(self.linear(forward_results["left_u_BNm1H"]))
        else:  # NOT backward variant
            intensity_B_Nm1_M = self._get_intensity(
                forward_results["left_xs_BNm1LP"], right_us_BNm1H
            )

        # sample dt in each interval for MC: [batch_size, num_times=N-1, num_mc_sample]
        # N-1 because we only consider the intervals between N events
        # G for grid points
        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(inter_event_times_BN[:, 1:])

        # evaluate intensity at dt_samples for MC *from the left limit* after decay -> shape (B, N-1, MC, M)
        intensity_dts_B_Nm1_G_M = self._evolve_and_get_intensity_at_sampled_dts(
            right_xs_BNLP[
                :, :-1
            ],  # x_{t_i+} will evolve up to x_{t_{i+1}-} and many times between for i=0,...,N-1
            dts_sample_B_Nm1_G,
            right_us_BNm1H,
        )

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
                lambda_at_event=intensity_B_Nm1_M,
                lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
                time_delta_seq=inter_event_times_BN[:, 1:],
                seq_mask=batch_non_pad_mask[:, 1:],
                type_seq=marks_BN[:, 1:],
        )

        # compute loss to optimize
        loss = -(event_ll - non_event_ll).sum()

        return loss, num_events

    def compute_intensities_at_sample_times(
        self, event_times_BN, inter_event_times_BN, marks_BN, sample_dtimes, **kwargs
    ):
        """Compute the intensity at sampled times, not only event times.  *from the left limit*

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get("compute_last_step_only", False)

        # assume inter_event_times_BN always starts from 0
        _input = event_times_BN, inter_event_times_BN, marks_BN, None, None

        # 'seq_len - 1' left limit for [t_1, ..., t_N]
        # 'seq_len' right limit for [t_0, t_1, ..., t_{N-1}, t_N]

        forward_results = self.forward(
            _input
        )  # N minus 1 comparing with sequence lengths
        right_xs_BNLP, right_us_BNH = (
            forward_results["right_xs_BNLP"],
            forward_results["right_us_BNH"],
        )

        if (
            compute_last_step_only
        ):  # fix indices for right_us_BNH: list [None, tensor([BNH]), ...]
            right_us_B1H = [
                None if right_u_BNH is None else right_u_BNH[:, -1:, :]
                for right_u_BNH in right_us_BNH
            ]
            sampled_intensity = self._evolve_and_get_intensity_at_sampled_dts(
                right_xs_BNLP[:, -1:, :, :], sample_dtimes[:, -1:, :], right_us_B1H
            )  # equiv. to right_xs_BNLP[:, -1, :, :][:, None, ...]
        else:
            sampled_intensity = self._evolve_and_get_intensity_at_sampled_dts(
                right_xs_BNLP, sample_dtimes, right_us_BNH
            )
        return sampled_intensity  # [B, N, MC, M]
