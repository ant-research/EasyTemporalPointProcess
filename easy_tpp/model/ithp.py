import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easy_tpp.model.basemodel import BaseModel


class _ITHPPositionwiseFeedForward(nn.Module):
    """Released ITHP feed-forward block with post-layer normalization."""

    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs):
        residual = inputs
        outputs = self.dropout(F.gelu(self.w_1(inputs)))
        outputs = self.dropout(self.w_2(outputs))
        return self.layer_norm(outputs + residual)


class _ITHPDynamicValueAttention(nn.Module):
    """ITHP attention without learned query or key projections."""

    def __init__(self, d_model, d_k, d_v, dropout):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.w_vs = nn.Linear(2 * d_model, d_v, bias=False)
        self.fc = nn.Linear(d_v, 2 * d_model)
        nn.init.xavier_uniform_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(2 * d_model, eps=1e-6)

    def project_values(self, source_inputs):
        return self.w_vs(source_inputs)

    def forward(self, query_inputs, source_inputs, source_values, allowed_mask):
        residual = query_inputs
        scores = torch.matmul(
            query_inputs / self.scale,
            source_inputs.transpose(-1, -2),
        )

        has_history = allowed_mask.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~allowed_mask, -torch.inf)
        masked_scores = torch.where(
            has_history,
            masked_scores,
            torch.zeros_like(masked_scores),
        )
        attention = F.softmax(masked_scores, dim=-1)
        attention = attention * allowed_mask.to(attention.dtype)
        attention = self.dropout(attention)

        outputs = torch.matmul(attention, source_values)
        outputs = self.dropout(self.fc(outputs))
        outputs = self.layer_norm(outputs + residual)
        return outputs, attention


class _ITHPEncoderLayer(nn.Module):
    """Single released ITHP attention and feed-forward layer."""

    def __init__(self, d_model, d_inner, d_k, d_v, dropout):
        super().__init__()
        self.self_attention = _ITHPDynamicValueAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )
        self.feed_forward = _ITHPPositionwiseFeedForward(
            d_model=d_v,
            d_inner=d_inner,
            dropout=dropout,
        )

    def forward(
        self,
        query_inputs,
        source_inputs,
        source_values,
        allowed_mask,
        query_mask,
    ):
        outputs, attention = self.self_attention(
            query_inputs=query_inputs,
            source_inputs=source_inputs,
            source_values=source_values,
            allowed_mask=allowed_mask,
        )
        outputs = self.feed_forward(outputs)
        outputs = outputs * query_mask.unsqueeze(-1).to(outputs.dtype)
        return outputs, attention


class ITHP(BaseModel):
    """Interpretable Transformer Hawkes Process.

    The architecture follows the KDD 2024 authors' public implementation:
    https://github.com/waystogetthere/Interpretable-Transformer-Hawkes-Process
    Paper: https://arxiv.org/abs/2405.16059
    Event indexing, padding, likelihood evaluation, and sampling are adapted
    to EasyTPP. Attention has one head by construction; ``num_heads`` is not
    used. Select integration with ``model_specs.integration_method`` so the
    choice survives EasyTPP's config serialization.

    Our reported runs used Adam epsilon 1e-5 and gradient-norm clipping at 1.
    EasyTPP's standard runner does not apply those settings, so fresh training
    here does not exactly reproduce that training path. Saved checkpoints load
    unchanged.
    """

    SUPPORTED_INTEGRATION_METHODS = {"mc", "trapezoid", "fixed_grid"}

    def __init__(self, model_config):
        super().__init__(model_config)

        specs = model_config.model_specs or {}
        self.d_model = int(model_config.hidden_size)
        self.d_inner = int(specs.get("d_inner", 128))
        self.d_k = int(specs.get("d_k", 16))
        self.d_v = int(specs.get("d_v", 2 * self.d_model))
        self.n_layers = int(model_config.num_layers)
        self.dropout = float(model_config.dropout_rate)

        self.integration_method = str(specs.get("integration_method", "mc")).lower()
        self.grid_step = float(specs.get("grid_step", 0.1))
        self.query_chunk_size = int(specs.get("query_chunk_size", 256))
        self.max_grid_points_per_interval = int(
            specs.get("max_grid_points_per_interval", 4096)
        )

        self.use_type_loss = bool(specs.get("use_type_loss", True))
        self.type_loss_weight = float(specs.get("type_loss_weight", 0.5))

        self._validate_configuration()
        if self.integration_method == "mc":
            self.use_mc_samples = True
        elif self.integration_method == "trapezoid":
            self.use_mc_samples = False

        position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (index // 2) / self.d_model)
                for index in range(self.d_model)
            ],
            dtype=torch.float32,
        )
        self.register_buffer("position_vec", position_vec)

        self.encoder_layer = _ITHPEncoderLayer(
            d_model=self.d_model,
            d_inner=self.d_inner,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=self.dropout,
        )
        self.intensity_decoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.d_v, 1), nn.Softplus())
                for _ in range(self.num_event_types)
            ]
        )
        self.type_predictor = nn.Linear(self.d_v, self.num_event_types)

        self.to(self.device)

    def _validate_configuration(self):
        if self.d_model <= 0 or self.d_model % 2:
            raise ValueError("ITHP hidden_size must be a positive even integer.")
        if self.d_inner <= 0 or self.d_k <= 0 or self.d_v <= 0:
            raise ValueError("ITHP d_inner, d_k, and d_v must be positive.")
        if self.n_layers != 1:
            raise ValueError("Released ITHP supports num_layers == 1.")
        if self.d_v != 2 * self.d_model:
            raise ValueError("Released ITHP requires d_v == 2 * hidden_size.")
        if self.integration_method not in self.SUPPORTED_INTEGRATION_METHODS:
            supported = ", ".join(sorted(self.SUPPORTED_INTEGRATION_METHODS))
            raise ValueError(
                f"Unsupported ITHP integration_method={self.integration_method!r}; "
                f"choose one of {supported}."
            )
        if self.loss_integral_num_sample_per_step <= 0:
            raise ValueError(
                "ITHP loss_integral_num_sample_per_step must be positive."
            )
        if (
            self.integration_method == "trapezoid"
            and self.loss_integral_num_sample_per_step < 2
        ):
            raise ValueError(
                "ITHP trapezoid integration requires at least two samples."
            )
        if self.grid_step <= 0:
            raise ValueError("ITHP grid_step must be positive.")
        if self.query_chunk_size <= 0:
            raise ValueError("ITHP query_chunk_size must be positive.")
        if self.max_grid_points_per_interval <= 0:
            raise ValueError(
                "ITHP max_grid_points_per_interval must be positive."
            )
        if self.type_loss_weight < 0:
            raise ValueError("ITHP type_loss_weight cannot be negative.")

    def make_dtime_loss_samples(self, time_delta_seq):
        """Draw independent uniform times for MC or fixed times for quadrature."""
        if self.use_mc_samples:
            ratios = torch.rand(
                (*time_delta_seq.shape, self.loss_integral_num_sample_per_step),
                device=self.device,
            )
        else:
            ratios = torch.linspace(
                0.0,
                1.0,
                self.loss_integral_num_sample_per_step,
                device=self.device,
            )[None, None, :]
        return time_delta_seq[..., None] * ratios

    def compute_temporal_embedding(self, times):
        """Encode absolute timestamps with the released sinusoidal features."""
        scaled_times = times.to(self.position_vec.dtype).unsqueeze(-1)
        scaled_times = scaled_times / self.position_vec
        embeddings = torch.empty_like(scaled_times)
        embeddings[..., 0::2] = torch.sin(scaled_times[..., 0::2])
        embeddings[..., 1::2] = torch.cos(scaled_times[..., 1::2])
        return embeddings

    def _compose_inputs(self, times, event_types, non_pad_mask):
        temporal = self.compute_temporal_embedding(times)
        type_embedding = self.layer_type_emb(event_types.long())
        inputs = torch.cat([temporal, type_embedding], dim=-1)
        return inputs * non_pad_mask.unsqueeze(-1).to(inputs.dtype)

    @staticmethod
    def _build_history_mask(source_mask, history_indices, query_mask):
        source_positions = torch.arange(
            source_mask.size(1),
            device=source_mask.device,
        ).view(1, 1, -1)
        allowed = source_positions <= history_indices.unsqueeze(-1)
        allowed = allowed & source_mask.unsqueeze(1)
        return allowed & query_mask.unsqueeze(-1)

    def _prepare_source(self, source_times, source_types, source_mask):
        source_inputs = self._compose_inputs(
            times=source_times,
            event_types=source_types,
            non_pad_mask=source_mask,
        )
        source_values = self.encoder_layer.self_attention.project_values(
            source_inputs
        )
        return source_inputs, source_values

    def _encode_query_chunk(
        self,
        source_inputs,
        source_values,
        source_mask,
        query_times,
        query_types,
        history_indices,
        query_mask,
    ):
        query_inputs = self._compose_inputs(
            times=query_times,
            event_types=query_types,
            non_pad_mask=query_mask,
        )
        allowed_mask = self._build_history_mask(
            source_mask=source_mask,
            history_indices=history_indices,
            query_mask=query_mask,
        )
        return self.encoder_layer(
            query_inputs=query_inputs,
            source_inputs=source_inputs,
            source_values=source_values,
            allowed_mask=allowed_mask,
            query_mask=query_mask,
        )

    def _marked_intensity_chunks(
        self,
        source_times,
        source_types,
        source_mask,
        query_times,
        history_indices,
        query_mask,
    ):
        source_inputs, source_values = self._prepare_source(
            source_times=source_times,
            source_types=source_types,
            source_mask=source_mask,
        )

        candidate_type_embeddings = self.layer_type_emb(
            torch.arange(
                self.num_event_types,
                dtype=torch.long,
                device=query_times.device,
            )
        )
        decoder_weights = torch.cat(
            [decoder[0].weight for decoder in self.intensity_decoders],
            dim=0,
        )
        decoder_biases = torch.cat(
            [decoder[0].bias for decoder in self.intensity_decoders],
            dim=0,
        )
        num_queries = query_times.size(1)
        for start in range(0, num_queries, self.query_chunk_size):
            end = min(start + self.query_chunk_size, num_queries)
            chunk_times = query_times[:, start:end]
            chunk_history = history_indices[:, start:end]
            chunk_mask = query_mask[:, start:end]

            batch_size, chunk_size = chunk_times.shape
            temporal_embeddings = self.compute_temporal_embedding(chunk_times)
            temporal_embeddings = temporal_embeddings.unsqueeze(1).expand(
                batch_size,
                self.num_event_types,
                chunk_size,
                self.d_model,
            )
            type_embeddings = candidate_type_embeddings.view(
                1,
                self.num_event_types,
                1,
                self.d_model,
            ).expand(
                batch_size,
                self.num_event_types,
                chunk_size,
                self.d_model,
            )
            query_inputs = torch.cat(
                [temporal_embeddings, type_embeddings],
                dim=-1,
            ).reshape(batch_size, self.num_event_types * chunk_size, -1)
            expanded_mask = chunk_mask.unsqueeze(1).expand(
                batch_size,
                self.num_event_types,
                chunk_size,
            ).reshape(batch_size, -1)
            query_inputs = (
                query_inputs * expanded_mask.unsqueeze(-1).to(query_inputs.dtype)
            )
            expanded_history = chunk_history.unsqueeze(1).expand(
                batch_size,
                self.num_event_types,
                chunk_size,
            ).reshape(batch_size, -1)
            allowed_mask = self._build_history_mask(
                source_mask=source_mask,
                history_indices=expanded_history,
                query_mask=expanded_mask,
            )
            enc_output, _ = self.encoder_layer(
                query_inputs=query_inputs,
                source_inputs=source_inputs,
                source_values=source_values,
                allowed_mask=allowed_mask,
                query_mask=expanded_mask,
            )
            enc_output = enc_output.reshape(
                batch_size,
                self.num_event_types,
                chunk_size,
                self.d_v,
            )

            logits = torch.einsum(
                "btqd,td->btq",
                enc_output,
                decoder_weights,
            )
            logits = logits + decoder_biases.view(1, -1, 1)
            chunk_intensities = F.softplus(logits).transpose(1, 2)
            chunk_intensities = (
                chunk_intensities
                * chunk_mask.unsqueeze(-1).to(chunk_intensities.dtype)
            )
            yield start, end, chunk_intensities

    def _compute_marked_intensities(
        self,
        source_times,
        source_types,
        source_mask,
        query_times,
        history_indices,
        query_mask,
    ):
        intensity_chunks = [
            intensities
            for _, _, intensities in self._marked_intensity_chunks(
                source_times=source_times,
                source_types=source_types,
                source_mask=source_mask,
                query_times=query_times,
                history_indices=history_indices,
                query_mask=query_mask,
            )
        ]
        if not intensity_chunks:
            return torch.empty(
                (*query_times.shape, self.num_event_types),
                dtype=self.layer_type_emb.weight.dtype,
                device=query_times.device,
            )
        return torch.cat(intensity_chunks, dim=1)

    def forward(self, time_seqs, type_seqs, batch_non_pad_mask):
        """Compute marked intensities for events after the first event."""
        source_times = time_seqs[:, :-1]
        source_types = type_seqs[:, :-1]
        source_mask = batch_non_pad_mask[:, :-1].bool()
        query_times = time_seqs[:, 1:]
        query_mask = batch_non_pad_mask[:, 1:].bool()
        history_indices = torch.arange(
            query_times.size(1),
            device=query_times.device,
        ).unsqueeze(0).expand(query_times.size(0), -1)

        return self._compute_marked_intensities(
            source_times=source_times,
            source_types=source_types,
            source_mask=source_mask,
            query_times=query_times,
            history_indices=history_indices,
            query_mask=query_mask,
        )

    def _compute_type_loss(self, time_seqs, type_seqs, batch_non_pad_mask):
        source_times = time_seqs[:, :-1]
        source_types = type_seqs[:, :-1]
        source_mask = batch_non_pad_mask[:, :-1].bool()
        query_mask = batch_non_pad_mask[:, 1:].bool()
        num_queries = source_times.size(1)
        history_indices = torch.arange(
            num_queries,
            device=source_times.device,
        ).unsqueeze(0).expand(source_times.size(0), -1) - 1

        source_inputs, source_values = self._prepare_source(
            source_times=source_times,
            source_types=source_types,
            source_mask=source_mask,
        )
        enc_output, _ = self._encode_query_chunk(
            source_inputs=source_inputs,
            source_values=source_values,
            source_mask=source_mask,
            query_times=source_times,
            query_types=source_types,
            history_indices=history_indices,
            query_mask=query_mask,
        )
        logits = self.type_predictor(enc_output)
        return F.cross_entropy(
            logits.transpose(1, 2),
            type_seqs[:, 1:],
            ignore_index=self.pad_token_id,
            reduction="sum",
        )

    def compute_intensities_at_sample_times(
        self,
        time_seqs,
        time_delta_seqs,
        type_seqs,
        sample_dtimes,
        **kwargs,
    ):
        """Compute marked intensities after each supplied source event."""
        del time_delta_seqs
        if sample_dtimes.dim() != 3:
            raise ValueError(
                "ITHP sample_dtimes must have shape [batch, sequence, samples]."
            )

        source_mask = kwargs.get("source_mask")
        if source_mask is None:
            source_mask = type_seqs.ne(self.pad_token_id)
        source_mask = source_mask.bool()

        interval_mask = kwargs.get("query_mask")
        if interval_mask is None:
            interval_mask = source_mask
        interval_mask = interval_mask.bool()

        batch_size, seq_len, num_samples = sample_dtimes.shape
        query_times = time_seqs.unsqueeze(-1) + sample_dtimes
        query_times = query_times.reshape(batch_size, seq_len * num_samples)
        history_indices = torch.arange(
            seq_len,
            device=time_seqs.device,
        ).view(1, seq_len, 1).expand(batch_size, seq_len, num_samples)
        history_indices = history_indices.reshape(batch_size, -1)
        query_mask = interval_mask.unsqueeze(-1).expand(
            batch_size,
            seq_len,
            num_samples,
        ).reshape(batch_size, -1)

        intensities = self._compute_marked_intensities(
            source_times=time_seqs,
            source_types=type_seqs,
            source_mask=source_mask,
            query_times=query_times,
            history_indices=history_indices,
            query_mask=query_mask,
        )
        intensities = intensities.reshape(
            batch_size,
            seq_len,
            num_samples,
            self.num_event_types,
        )

        if kwargs.get("compute_last_step_only", False):
            return intensities[:, -1:, :, :]
        return intensities

    def _integrate_fixed_grid(
        self,
        source_times,
        source_types,
        source_mask,
        time_delta_seqs,
        interval_mask,
    ):
        cell_counts = torch.ceil(time_delta_seqs / self.grid_step).long()
        cell_counts = torch.where(
            interval_mask & time_delta_seqs.gt(0),
            cell_counts,
            torch.zeros_like(cell_counts),
        )
        max_cells = int(cell_counts.max().item()) if cell_counts.numel() else 0
        if max_cells > self.max_grid_points_per_interval:
            raise ValueError(
                "ITHP fixed_grid requires "
                f"{max_cells} points in one interval, exceeding "
                f"max_grid_points_per_interval="
                f"{self.max_grid_points_per_interval}. Use integration_method="
                "'mc' or increase grid_step."
            )
        if max_cells == 0:
            return torch.zeros_like(time_delta_seqs)

        batch_size, num_intervals = time_delta_seqs.shape
        cell_index = torch.arange(
            max_cells,
            device=time_delta_seqs.device,
            dtype=time_delta_seqs.dtype,
        ).view(1, 1, -1)
        cell_start = cell_index * self.grid_step
        cell_width = torch.clamp(
            time_delta_seqs.unsqueeze(-1) - cell_start,
            min=0.0,
            max=self.grid_step,
        )
        cell_mask = interval_mask.unsqueeze(-1) & cell_width.gt(0)
        query_times = (
            source_times.unsqueeze(-1)
            + cell_start
            + 0.5 * cell_width
        )

        flat_query_times = query_times.reshape(batch_size, -1)
        flat_query_mask = cell_mask.reshape(batch_size, -1)
        flat_weights = cell_width.reshape(batch_size, -1)
        flat_history = torch.arange(
            num_intervals,
            device=time_delta_seqs.device,
        ).view(1, num_intervals, 1).expand(
            batch_size,
            num_intervals,
            max_cells,
        ).reshape(batch_size, -1)
        flat_intervals = torch.arange(
            num_intervals,
            device=time_delta_seqs.device,
        ).view(1, num_intervals, 1).expand(
            batch_size,
            num_intervals,
            max_cells,
        ).reshape(batch_size, -1)

        non_event_ll = torch.zeros_like(time_delta_seqs)
        for start, end, intensities in self._marked_intensity_chunks(
            source_times=source_times,
            source_types=source_types,
            source_mask=source_mask,
            query_times=flat_query_times,
            history_indices=flat_history,
            query_mask=flat_query_mask,
        ):
            weighted_intensity = (
                intensities.sum(dim=-1) * flat_weights[:, start:end]
            )
            interval_indices = flat_intervals[:, start:end]
            contribution = torch.zeros_like(non_event_ll).scatter_add(
                dim=1,
                index=interval_indices,
                src=weighted_intensity,
            )
            non_event_ll = non_event_ll + contribution
        return non_event_ll

    def _compute_grid_loglikelihood(
        self,
        time_delta_seqs,
        lambda_at_event,
        non_event_ll,
        seq_mask,
        type_seqs,
    ):
        lambda_at_event = lambda_at_event + self.eps
        log_marked_event_lambdas = lambda_at_event.log()
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),
            target=type_seqs,
            ignore_index=self.pad_token_id,
            reduction="none",
        )
        non_event_ll = non_event_ll * seq_mask * time_delta_seqs.gt(0)
        num_events = int(seq_mask.sum().item())
        return event_ll, non_event_ll, num_events

    def loglike_loss(self, batch=None, **kwargs):
        """Compute ITHP likelihood and optional auxiliary type loss."""
        (
            time_seqs,
            time_delta_seqs,
            type_seqs,
            batch_non_pad_mask,
            _,
        ) = self.resolve_batch_inputs(batch, kwargs)
        if time_seqs.size(1) < 2:
            raise ValueError("ITHP requires sequences with at least two events.")

        target_mask = batch_non_pad_mask[:, 1:].bool()
        lambda_at_event = self.forward(
            time_seqs=time_seqs,
            type_seqs=type_seqs,
            batch_non_pad_mask=batch_non_pad_mask,
        )

        source_times = time_seqs[:, :-1]
        source_types = type_seqs[:, :-1]
        source_mask = batch_non_pad_mask[:, :-1].bool()
        target_dtimes = time_delta_seqs[:, 1:]
        target_types = type_seqs[:, 1:]

        if self.integration_method == "fixed_grid":
            non_event_ll = self._integrate_fixed_grid(
                source_times=source_times,
                source_types=source_types,
                source_mask=source_mask,
                time_delta_seqs=target_dtimes,
                interval_mask=target_mask,
            )
            event_ll, non_event_ll, num_events = self._compute_grid_loglikelihood(
                time_delta_seqs=target_dtimes,
                lambda_at_event=lambda_at_event,
                non_event_ll=non_event_ll,
                seq_mask=target_mask,
                type_seqs=target_types,
            )
        else:
            sample_dtimes = self.make_dtime_loss_samples(target_dtimes)
            lambda_t_sample = self.compute_intensities_at_sample_times(
                time_seqs=source_times,
                time_delta_seqs=time_delta_seqs[:, :-1],
                type_seqs=source_types,
                sample_dtimes=sample_dtimes,
                source_mask=source_mask,
                query_mask=target_mask,
            )
            (event_ll, non_event_ll, num_events) = self.compute_loglikelihood(
                time_delta_seq=target_dtimes,
                lambda_at_event=lambda_at_event,
                lambdas_loss_samples=lambda_t_sample,
                seq_mask=target_mask,
                type_seq=target_types,
            )

        if num_events == 0:
            raise ValueError("ITHP batch contains no target events.")

        nll = -(event_ll - non_event_ll).sum()
        loss = nll
        if self.training and self.use_type_loss and self.type_loss_weight > 0:
            type_loss = self._compute_type_loss(
                time_seqs=time_seqs,
                type_seqs=type_seqs,
                batch_non_pad_mask=batch_non_pad_mask,
            )
            loss = loss + self.type_loss_weight * type_loss

        return loss, num_events
