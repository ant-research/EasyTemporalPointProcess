import torch
from torch import nn
import math

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

class RMTPP(TorchBaseModel):
    """Torch implementation of Recurrent Marked Temporal Point Processes, KDD 2016.
    https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(RMTPP, self).__init__(model_config)

        self.layer_temporal_emb = nn.Linear(1, self.hidden_size)
        self.layer_rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                num_layers=1, batch_first=True)

        self.hidden_to_intensity_logits = nn.Linear(self.hidden_size, self.num_event_types)
        self.b_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.w_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        nn.init.xavier_normal_(self.b_t)
        nn.init.xavier_normal_(self.w_t)

    def evolve_and_get_intentsity(self, right_hiddens_BNH, dts_BNG):
        """
        Eq.11 that computes intensity.
        """
        past_influence_BNGM = self.hidden_to_intensity_logits(right_hiddens_BNH[..., None, :])
        intensity_BNGM = (past_influence_BNGM + self.w_t[None, None, :] * dts_BNG[..., None]
                         + self.b_t[None, None, :]).clamp(max=math.log(1e5)).exp()
        return intensity_BNGM

    def forward(self, batch):
        """
        Suppose we have inputs with original sequence length N+1
        ts: [t0, t1, ..., t_N]
        dts: [0, t1 - t0, t2 - t1, ..., t_N - t_{N-1}]
        marks: [k0, k1, ..., k_N] (k0 and kN could be padded marks if t0 and tN correspond to left and right windows)

        Return:
            left limits of *intensity* at [t_1, ..., t_N] of shape: (batch_size, seq_len - 1, hidden_dim)
            right limits of *hidden states* [t_0, ..., t_{N-1}, t_N] of shape: (batch_size, seq_len, hidden_dim)
            We need the right limit of t_N to sample continuation.
        """

        t_BN, dt_BN, marks_BN, _, _ = batch
        mark_emb_BNH = self.layer_type_emb(marks_BN)
        time_emb_BNH = self.layer_temporal_emb(t_BN[..., None])
        right_hiddens_BNH, _ = self.layer_rnn(mark_emb_BNH + time_emb_BNH)
        left_intensity_B_Nm1_M = self.evolve_and_get_intentsity(right_hiddens_BNH[:, :-1, :], dt_BN[:, 1:][...,None]).squeeze(-2)
        return left_intensity_B_Nm1_M, right_hiddens_BNH


    def loglike_loss(self, batch):
        """Compute the log-likelihood loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch

        # compute left intensity and hidden states at event time
        # left limits of intensity at [t_1, ..., t_N]
        # right limits of hidden states at [t_0, ..., t_{N-1}, t_N]
        left_intensity_B_Nm1_M, right_hiddens_BNH = self.forward((ts_BN, dts_BN, marks_BN, None, None))
        right_hiddens_B_Nm1_H = right_hiddens_BNH[..., :-1, :]  # discard right limit at t_N for logL

        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(dts_BN[:, 1:])
        intensity_dts_B_Nm1_G_M = self.evolve_and_get_intentsity(right_hiddens_B_Nm1_H, dts_sample_B_Nm1_G)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=left_intensity_B_Nm1_M,
            lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
            time_delta_seq=dts_BN[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=marks_BN[:, 1:]
        )

        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events



    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        _input = time_seqs, time_delta_seqs, type_seqs, None, None
        _, right_hiddens_BNH = self.forward(_input)

        if compute_last_step_only:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH[:, -1:, :], sample_dtimes[:, -1:, :])
        else:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH, sample_dtimes)  # shape: [B, N, G, M]
        return sampled_intensities
