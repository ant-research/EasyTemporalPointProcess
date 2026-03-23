"""Weighted Score Matching for Temporal Point Processes (WSM-TPP).

Implements a THP-style causal Transformer trained with a weighted score
matching objective plus mark cross-entropy.

Training objective:
    L = L_WSM + CE_coef * L_CE (+ optional survival loss)

EasyTPP integration notes:
- train phase uses the WSM objective
- validation/test phases use approximate log-likelihood via nll_loss
- prediction uses compute_intensities_at_sample_times
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_baselayer import MultiHeadAttention


# Internal helpers (module-level, not exported)
class _EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with multi-head self-attention."""

    def __init__(self, d_model: int, d_inner: int, n_head: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Reuse MultiHeadAttention from torch_baselayer.
        self.self_attn = MultiHeadAttention(
            n_head=n_head,
            d_input=d_model,
            d_model=d_model,
            dropout=dropout,
            output_linear=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual attention
        x_n = self.norm1(x)
        x = x + self.self_attn(x_n, x_n, x_n, attn_mask)
        # Pre-norm + residual FFN
        x = x + self.ffn(self.norm2(x))
        return x


# Main model
class WSMTHP(TorchBaseModel):
    """THP encoder trained with Weighted Score Matching (WSM).

    The class name ``WSMTHP`` must match the ``model_id`` field in the YAML config.

    YAML ``model_config`` parameters
    ----------------------------------
    Standard (top-level):
        hidden_size            int    Transformer embedding / model dimension.
        num_layers             int    Number of Transformer encoder layers.
        num_heads              int    Number of attention heads.
        dropout_rate           float  Dropout probability.

    WSM-specific (under ``model_specs``):
        d_inner          int    FFN inner dimension (default: hidden_size * 2).
        CE_coef          float  Weight for the mark cross-entropy term (default: 10.0).
        h_type           str    Weight function selector; currently only
                                ``'two_side_op'`` is supported (default).
        T_mode           str    How to obtain the observation window end T:
                                ``'manual'``, ``'train_global'``, or ``'batch'``.
        max_observed_time float  Observation window end T used when ``T_mode`` is
                                 ``'manual'``. When omitted under ``'batch'``,
                                 T = max(time_seqs) inside each batch.
    """

    def __init__(self, model_config):
        super(WSMTHP, self).__init__(model_config)

        d_model = self.hidden_size
        specs = model_config.model_specs  # dict of WSM-specific hyper-params

        d_inner = specs.get('d_inner', d_model * 2)
        n_head = model_config.num_heads
        n_layers = model_config.num_layers
        dropout = model_config.dropout_rate

        self.h_type = specs.get('h_type', 'two_side_op')
        self.CE_coef = float(specs.get('CE_coef', 10.0))
        self.T_mode = str(specs.get('T_mode', 'train_global')).lower()
        # T: observation window end. None means batch-wise T at runtime.
        self.max_observed_time = specs.get('max_observed_time', None)
        if self.max_observed_time is not None:
            self.max_observed_time = float(self.max_observed_time)
        if self.T_mode == 'manual' and self.max_observed_time is None:
            raise ValueError('WSMTHP with T_mode=manual requires max_observed_time.')

        #  Temporal positional encoding 
        self.register_buffer(
            '_pos_vec',
            torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)]
            )
        )

        #  Causal Transformer encoder 
        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(d_model, d_inner, n_head, dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        #  Intensity head: _m(|h) = softplus(tanh(aff_m(h)) + base_m(h)) 
        self.affect_layer = nn.Linear(d_model, self.num_event_types)
        self.base_layer = nn.Linear(d_model, self.num_event_types)
        nn.init.xavier_normal_(self.affect_layer.weight)
        nn.init.xavier_normal_(self.base_layer.weight)
        nn.init.zeros_(self.affect_layer.bias)
        nn.init.zeros_(self.base_layer.bias)
        self.intensity_act = nn.Softplus(beta=1.0)

        self.with_survival = bool(specs.get('with_survival', False))
        self.alpha_survival = float(specs.get('alpha_survival', 50.0))
        self.alpha_neg = float(specs.get('alpha_neg', 50.0))
        if self.with_survival:
            self.survival_head = nn.Sequential(
                nn.Linear(d_model, d_inner),
                nn.ReLU(),
                nn.Linear(d_inner, 1),
            )
        else:
            self.alpha_survival = 0.0
            self.survival_head = None


        self.to(self.device)

    #  Private helpers 

    def _temporal_enc(self, time_seqs: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of absolute event times.

        Args:
            time_seqs: [B, N]
        Returns:
            [B, N, d_model]
        """
        t = time_seqs.unsqueeze(-1) / self._pos_vec  # [B, N, d]
        enc = torch.zeros_like(t)
        enc[..., 0::2] = torch.sin(t[..., 0::2])
        enc[..., 1::2] = torch.cos(t[..., 1::2])
        return enc

    def _causal_attn_mask(self, type_seqs: torch.Tensor) -> torch.Tensor:
        """Build combined causal + key-padding attention mask.

        Convention (same as torch_baselayer.MultiHeadAttention):
            1 = masked,  0 = attended.

        Args:
            type_seqs: [B, N] long
        Returns:
            [B, N, N] uint8
        """
        B, N = type_seqs.shape
        device = type_seqs.device

        # Upper-triangular causal mask (future positions)
        causal = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.uint8), diagonal=1
        ).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # Key-padding mask (PAD tokens should not be attended to)
        pad_mask = type_seqs.eq(self.pad_token_id).unsqueeze(1).expand(-1, N, -1)  # [B, N, N]

        return (causal + pad_mask.to(torch.uint8)).gt(0).to(torch.uint8)

    def _encode(
        self,
        type_seqs: torch.Tensor,
        time_seqs: torch.Tensor,
        non_pad_mask_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Causal Transformer encoder pass.

        Args:
            type_seqs:       [B, N] long
            time_seqs:       [B, N] float
            non_pad_mask_3d: [B, N, 1] float (1 = real event, 0 = padded)
        Returns:
            [B, N, d_model]
        """
        type_emb = self.layer_type_emb(type_seqs)            # [B, N, d]
        tem_enc = self._temporal_enc(time_seqs)               # [B, N, d]
        x = (type_emb + tem_enc) * non_pad_mask_3d           # zero-out pad positions

        attn_mask = self._causal_attn_mask(type_seqs)        # [B, N, N]
        for layer in self.encoder_layers:
            x = layer(x, attn_mask) * non_pad_mask_3d

        x = self.encoder_norm(x) * non_pad_mask_3d
        return x

    def _get_intensity(
        self,
        tau: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-mark intensities.

        Args:
            tau:  [B, N] or [B, N, S] inter-event times
            cond: [B, N, d_model]     history encoding
        Returns:
            [B, N, M] when tau is 2-D, or [B, N, S, M] when tau is 3-D
        """
        squeeze = (tau.ndim == 2)
        if tau.ndim == 2:
            tau = tau.unsqueeze(2)                             # [B, N, 1]

        affect = torch.tanh(self.affect_layer(cond)).unsqueeze(2)  # [B, N, 1, M]
        base = self.base_layer(cond).unsqueeze(2)                  # [B, N, 1, M]
        pre_lambda = affect * tau.unsqueeze(3) + base              # [B, N, S, M]
        intensity = self.intensity_act(pre_lambda)

        if squeeze:
            intensity = intensity.squeeze(2)                   # [B, N, M]
        return intensity

    def _compute_score(
        self,
        tau_var: torch.Tensor,
        cond: torch.Tensor,
        mask_2d: torch.Tensor,
    ):
        """Compute total intensity and score at tau_var.

        Score: s() = ?log _tot() / ? ? _tot()

        Args:
            tau_var:  [B, N] requires_grad=True
            cond:     [B, N, d_model]
            mask_2d:  [B, N] float (1 = real event, 0 = pad)
        Returns:
            all_intensity: [B, N, M]
            score:         [B, N]
        """
        all_intensity = self._get_intensity(tau_var, cond)          # [B, N, M]
        all_intensity = all_intensity * mask_2d.unsqueeze(-1)        # mask padding

        lambda_tot = all_intensity.sum(-1)                           # [B, N]
        log_lambda_tot = (lambda_tot + self.eps).log() * mask_2d

        grad = torch.autograd.grad(
            log_lambda_tot.sum(), tau_var, create_graph=True
        )[0] * mask_2d

        score = grad - lambda_tot
        return all_intensity, score

    def _wsm_weights(
        self,
        t_prior: torch.Tensor,
        t_curr: torch.Tensor,
        mask: torch.Tensor,
        T: float,
    ):
        """Compute weight function h and its derivative h' for two_side_op.

        h(_n)  = (T ?t_{n?})/2  ? |t_n ?(T + t_{n?})/2|
        h'(_n) = ?  if t_n > (T + t_{n?})/2,  else +1

        Args:
            t_prior: [B, N] absolute times t_{n-1}
            t_curr:  [B, N] absolute times t_n
            mask:    [B, N] float
            T:       float, observation window end
        Returns:
            h, hprime: [B, N] each
        """
        midpoint = (T + t_prior) / 2.0
        h = (T - t_prior) / 2.0 - torch.abs(t_curr - midpoint)
        h = h * mask

        hprime = torch.where(
            t_curr > midpoint,
            torch.full_like(t_curr, -1.0),
            torch.ones_like(t_curr),
        ) * mask
        return h, hprime

    #  EasyTPP public interface 

    def _compute_non_event_integral(
        self,
        lambdas_loss_samples: torch.Tensor,
        time_delta_seq: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate the integrated total intensity over each interval."""
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)
        if self.use_mc_samples:
            return total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        return 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

    def _get_survival_labels(self, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """Return labels indicating whether an event is the terminal event in the sequence."""
        labels = torch.zeros_like(non_pad_mask, dtype=torch.long)
        seq_lens = non_pad_mask.long().sum(dim=1)
        valid_batch = seq_lens > 0
        if valid_batch.any():
            labels[valid_batch, seq_lens[valid_batch] - 1] = 1
        return labels

    def _get_survival_continue_labels(self, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """Return labels for the continuation probability F_n.

        1 means the process continues after the current history, and 0 means the
        current history is terminal.
        """
        terminal = self._get_survival_labels(non_pad_mask)
        return (1 - terminal).float() * non_pad_mask.float()

    def _get_survival_loss(self, enc_out: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """Binary classification loss for the continuation probability F_n."""
        if not self.with_survival:
            return enc_out.new_zeros(())

        valid = non_pad_mask.bool()
        if not valid.any():
            return enc_out.new_zeros(())

        logits = self.survival_head(enc_out).squeeze(-1)[valid]
        labels = self._get_survival_continue_labels(non_pad_mask)[valid]
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        if self.alpha_neg != 1.0:
            continue_weight = torch.full_like(bce, self.alpha_neg)
            bce = bce * torch.where(labels > 0.5, continue_weight, torch.ones_like(bce))
        return bce.mean()

    def _get_survival_logits(self, enc_out: torch.Tensor) -> torch.Tensor:
        """Raw logits for the continuation probability F_n."""
        if not self.with_survival:
            raise RuntimeError('Survival logits requested when with_survival is disabled.')
        return self.survival_head(enc_out).squeeze(-1)

    def _get_survival_continue_prob(self, enc_out: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """Probability that the process continues before T after each event."""
        if not self.with_survival:
            return enc_out.new_zeros(non_pad_mask.shape)
        logits = self._get_survival_logits(enc_out)
        probs = torch.sigmoid(logits)
        return probs * non_pad_mask.float()

    def _get_survival_temporal_log_prob(self, enc_out: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """Conditional-on-first-event survival log-probability.

        This matches the EasyTPP convention that event likelihood terms are computed
        from histories H_1, ..., H_{N-1} for events 2, ..., N. The survival part is
        therefore
            sum_{n=2}^N log F_n(H_{n-1}) + log(1 - F_{N+1}(H_N)).
        """
        if not self.with_survival:
            return enc_out.new_zeros(())

        logits = self._get_survival_logits(enc_out)
        valid = non_pad_mask.bool()
        if not valid.any():
            return enc_out.new_zeros(())

        seq_lens = valid.long().sum(dim=1)
        valid_batch = seq_lens > 0

        continue_mask = valid.clone()
        continue_mask[valid_batch, seq_lens[valid_batch] - 1] = False
        continue_log_prob = F.logsigmoid(logits) * continue_mask.float()
        continue_log_prob = continue_log_prob.sum()

        terminal_log_prob = enc_out.new_zeros(())
        if valid_batch.any():
            batch_idx = torch.arange(enc_out.size(0), device=enc_out.device)[valid_batch]
            last_idx = seq_lens[valid_batch] - 1
            terminal_logits = logits[batch_idx, last_idx]
            terminal_log_prob = F.logsigmoid(-terminal_logits).sum()

        return continue_log_prob + terminal_log_prob

    def _compute_integral_to_samples(
        self,
        sample_dtimes: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate the integrated total intensity up to each sampled time."""
        if sample_dtimes.ndim != 3:
            raise ValueError('sample_dtimes must have shape [B, N, S].')

        batch_size, seq_len, num_sample = sample_dtimes.shape
        num_grid = self.loss_integral_num_sample_per_step

        ratios = torch.linspace(
            start=0.0,
            end=1.0,
            steps=num_grid,
            device=sample_dtimes.device,
        )[None, None, None, :]

        sampled_grid = sample_dtimes.unsqueeze(-1) * ratios
        sampled_grid_flat = sampled_grid.permute(0, 2, 1, 3).reshape(batch_size * num_sample, seq_len, num_grid)
        cond_flat = cond.unsqueeze(1).expand(-1, num_sample, -1, -1).reshape(batch_size * num_sample, seq_len, cond.size(-1))

        lambdas = self._get_intensity(sampled_grid_flat, cond_flat)
        total_lambdas = lambdas.sum(dim=-1).reshape(batch_size, num_sample, seq_len, num_grid).permute(0, 2, 1, 3)

        if self.use_mc_samples:
            return total_lambdas.mean(dim=-1) * sample_dtimes

        return 0.5 * (total_lambdas[..., 1:] + total_lambdas[..., :-1]).mean(dim=-1) * sample_dtimes


    def forward(self, batch):
        """Run the encoder on a batch.

        Args:
            batch: tuple (time_seqs, time_delta_seqs, type_seqs, non_pad_mask, attn_mask)
        Returns:
            enc_output: [B, N, d_model]
        """
        time_seqs, _, type_seqs, non_pad_mask, _ = batch
        return self._encode(type_seqs, time_seqs, non_pad_mask.float().unsqueeze(-1))

    def nll_loss(self, batch):
        """Compute approximate negative log-likelihood for evaluation."""
        time_seqs, time_delta_seqs, type_seqs, non_pad_mask, _ = batch

        non_pad_mask_f = non_pad_mask.float()
        non_pad_mask_3d = non_pad_mask_f.unsqueeze(-1)

        enc_out = self._encode(type_seqs, time_seqs, non_pad_mask_3d)
        cond = enc_out[:, :-1]
        tau = time_delta_seqs[:, 1:]
        mask = non_pad_mask_f[:, 1:]

        lambda_at_event = self._get_intensity(tau, cond)
        sample_dtimes = self.make_dtime_loss_samples(tau)
        lambda_t_sample = self._get_intensity(sample_dtimes, cond)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=tau,
            seq_mask=mask,
            type_seq=type_seqs[:, 1:],
        )

        ll = (event_ll - non_event_ll).sum()
        if self.with_survival:
            T = self.max_observed_time if self.max_observed_time is not None else float(time_seqs.max().item())
            time_to_end = (T - time_seqs[:, :-1]).clamp_min(0.0)
            lambda_t_end = self._get_intensity(self.make_dtime_loss_samples(time_to_end), cond)
            integral_to_end = self._compute_non_event_integral(lambda_t_end, time_to_end, mask)
            continue_logprob = torch.log((1.0 - torch.exp(-integral_to_end)).clamp_min(self.eps)) * mask
            survival_logprob = self._get_survival_temporal_log_prob(enc_out, non_pad_mask_f)
            ll = ll - continue_logprob.sum() + survival_logprob

        loss = -ll
        return loss, num_events

    def loglike_loss(self, batch):
        """Compute the WSM training loss.

        Required by EasyTPP's TorchModelWrapper.run_batch().
        Despite the method name, this returns the Weighted Score Matching
        objective (NOT log-likelihood). See module docstring for details.

        Args:
            batch: tuple unpacked from a BatchEncoding
                   (time_seqs, time_delta_seqs, type_seqs, non_pad_mask, attn_mask)
        Returns:
            loss:       scalar Tensor (WSM + CE mark loss)
            num_events: int, number of real (non-padded) events in the batch
        """
        if not self.training:
            return self.nll_loss(batch)

        time_seqs, time_delta_seqs, type_seqs, non_pad_mask, _ = batch

        non_pad_mask_f = non_pad_mask.float()              # [B, N]
        non_pad_mask_3d = non_pad_mask_f.unsqueeze(-1)    # [B, N, 1]
        # Encode history
        # enc_out[:, i] encodes events 0..i (causal).
        # Use enc_out[:, :-1] to predict events 1..N.
        enc_out = self._encode(type_seqs, time_seqs, non_pad_mask_3d)   # [B, N, d]

        cond = enc_out[:, :-1]                           # [B, N-1, d]
        tau = time_delta_seqs[:, 1:]                     # [B, N-1]  inter-event times _n
        mask = non_pad_mask_f[:, 1:]                     # [B, N-1]
        t_prior = time_seqs[:, :-1]                      # [B, N-1]  t_{n-1}
        t_curr = time_seqs[:, 1:]                        # [B, N-1]  t_n
        type_targets = type_seqs[:, 1:]                  # [B, N-1]
        # Observation window end T
        T = self.max_observed_time if self.max_observed_time is not None \
            else float(time_seqs.max().item())
        # Compute score and score derivative
        # Autograd requires tau with requires_grad=True; detach first.
        tau_var = tau.detach().requires_grad_(True)

        all_intensity, score = self._compute_score(tau_var, cond, mask)

        score_grad = torch.autograd.grad(
            score.sum(), tau_var,
            create_graph=True,
            retain_graph=True,
        )[0]
        # Weight function h and h'
        h, hprime = self._wsm_weights(t_prior, t_curr, mask, T)
        # WSM loss
        wsm_loss = (0.5 * h * score ** 2 + score_grad * h + score * hprime) * mask
        # Mark cross-entropy loss
        # CE = -log lambda_k(t_n) + log sum_m lambda_m(t_n)
        sum_intensity = all_intensity.sum(-1)             # [B, N-1]
        # Clamp type index: padded positions get index 0, but are zeroed by mask.
        type_idx = type_targets.long().clamp(0, self.num_event_types - 1)
        type_one_hot = F.one_hot(type_idx, self.num_event_types).float()
        type_intensity = (all_intensity * type_one_hot).sum(-1)       # [B, N-1]

        ce_loss = (
            -(type_intensity + self.eps).log()
            + (sum_intensity + self.eps).log()
        ) * mask
        # Survival loss is added below when with_survival is enabled.
        # See the paper for the finite-window survival correction.
        # Total loss
        if self.with_survival:
            survival_loss = self._get_survival_loss(enc_out, non_pad_mask_f)
        else:
            survival_loss = enc_out.new_zeros(())

        loss = (wsm_loss + self.CE_coef * ce_loss).sum() + self.alpha_survival * survival_loss
        num_events = int(mask.sum().item())

        return loss, num_events

    def compute_intensities_at_sample_times(
        self,
        time_seqs,
        time_delta_seqs,
        type_seqs,
        sample_dtimes,
        **kwargs,
    ):
        """Compute intensities at sampled inter-event times (for thinning-based generation).

        Called by the EventSampler in TorchBaseModel for prediction.

        Args:
            time_seqs:       [B, N]
            time_delta_seqs: [B, N]
            type_seqs:       [B, N]
            sample_dtimes:   [B, N, S]
        Returns:
            [B, N, S, M] or [B, 1, S, M] intensities
        """
        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        non_pad_mask_3d = (type_seqs != self.pad_token_id).float().unsqueeze(-1)
        enc_out = self._encode(type_seqs, time_seqs, non_pad_mask_3d)

        if compute_last_step_only:
            cond = enc_out[:, -1:, :]
            dts = sample_dtimes[:, -1:, :]
            hist_times = time_seqs[:, -1:]
            hist_mask = non_pad_mask_3d[:, -1:, 0]
        else:
            cond = enc_out
            dts = sample_dtimes
            hist_times = time_seqs
            hist_mask = non_pad_mask_3d[..., 0]

        raw_intensity = self._get_intensity(dts, cond)
        if not self.with_survival:
            return raw_intensity

        T = self.max_observed_time if self.max_observed_time is not None else float(time_seqs.max().item())
        time_to_end = (T - hist_times).clamp_min(0.0) * hist_mask

        integral_to_dt = self._compute_integral_to_samples(dts, cond) * hist_mask.unsqueeze(-1)
        g_dt = torch.exp(-integral_to_dt)

        lambda_t_end = self._get_intensity(self.make_dtime_loss_samples(time_to_end), cond)
        integral_to_end = self._compute_non_event_integral(lambda_t_end, time_to_end, hist_mask)
        g_t = torch.exp(-integral_to_end).unsqueeze(-1)

        continue_prob = self._get_survival_continue_prob(cond, hist_mask).clamp_min(self.eps).unsqueeze(-1)
        denom = g_dt + ((1.0 - g_t) / continue_prob) - 1.0
        corrected_intensity = (g_dt / denom.clamp_min(self.eps)).unsqueeze(-1) * raw_intensity

        within_horizon = (dts <= (time_to_end.unsqueeze(-1) + self.eps)).float().unsqueeze(-1)
        return corrected_intensity * within_horizon * hist_mask.unsqueeze(-1).unsqueeze(-1)
