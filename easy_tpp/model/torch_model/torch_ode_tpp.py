import torch
from torch import nn

from easy_tpp.model.torch_model.torch_baselayer import DNN
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.utils import rk4_step_method


def flatten_parameters(model):
    p_shapes = []
    flat_parameters = []
    for p in model.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)


class NeuralODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_init, delta_t, ode_fn, solver, num_sample_times, *model_parameters):
        """

        Args:
            ctx:
            input: (tensor): [batch_size]
            model:
            solver:
            delta_t (tensor): [batch_size, num_sample_times]

        Returns:

        """

        ctx.ode_fn = ode_fn
        ctx.solver = solver
        ctx.delta_t = delta_t
        ctx.model_parameters = model_parameters
        ctx.num_sample_times = num_sample_times

        total_state = []
        dt_ratio = 1.0 / num_sample_times
        delta_t = delta_t * dt_ratio
        with torch.no_grad():
            state = z_init
            for i in range(num_sample_times):
                # [batch_size, hidden_size]
                state = solver(diff_func=ode_fn, dt=delta_t, z0=state)
                total_state.append(state)

        # [batch_size, num_samples, hidden_size]
        ctx.save_for_backward(state)

        return state

    @staticmethod
    def backward(ctx, grad_z):
        output_state = ctx.saved_tensors[0]  # return a tuple
        ode_fn = ctx.ode_fn
        solver = ctx.solver
        delta_t = ctx.delta_t
        model_parameters = ctx.model_parameters
        num_sample_times = ctx.num_sample_times

        # Dynamics of augmented system to be calculated backwards in time
        def aug_dynamics(aug_states):
            tmp_z = aug_states[0]
            tmp_neg_a = -aug_states[1]

            with torch.set_grad_enabled(True):
                tmp_z = tmp_z.detach().requires_grad_(True)
                func_eval = ode_fn(tmp_z)
                tmp_ds = torch.autograd.grad(
                    (func_eval,), (tmp_z, *model_parameters),
                    grad_outputs=tmp_neg_a,
                    allow_unused=True,
                    retain_graph=True)

            neg_adfdz = tmp_ds[0]
            neg_adfdtheta = [torch.flatten(var) for var in tmp_ds[1:]]

            return [func_eval, neg_adfdz, *neg_adfdtheta]

        dt_ratio = 1.0 / num_sample_times
        delta_t = delta_t * dt_ratio

        with torch.no_grad():
            # Construct back-state for ode solver
            # reshape variable \theta for batch solving
            init_var_grad = [torch.zeros_like(torch.flatten(var)) for var in model_parameters]

            # [z(t_1), a(t_1), \theta]
            z1 = output_state
            a1 = grad_z
            states = [z1, a1, *init_var_grad]

            for i in range(num_sample_times):
                states = solver(aug_dynamics, -delta_t, states)

            grad_z0 = states[1]

            grad_theta = [torch.reshape(torch.mean(var_grad, dim=0), var.shape) for var, var_grad in
                          zip(model_parameters, states[2:])]

        return (grad_z0, None, None, None, None, *grad_theta)


class NeuralODE(nn.Module):
    def __init__(self, model, solver, num_sample_times):
        super().__init__()
        self.model = model
        self.solver = solver
        self.params = [w for w in model.parameters()]
        self.num_sample_times = num_sample_times

    def forward(self, input_state, delta_time):
        """

        Args:
            input_state: [batch_size, hidden_size]
            return_state:

        Returns:

        """
        output_state = NeuralODEAdjoint.apply(input_state,
                                              delta_time,
                                              self.model,
                                              self.solver,
                                              self.num_sample_times,
                                              *self.params)

        # [batch_size, num_sample_times, hidden_size]
        return output_state


class ODETPP(TorchBaseModel):
    """Torch implementation of a TPP with Neural ODE state evolution, which is a simplified version of TPP in
    https://arxiv.org/abs/2011.04583, ICLR 2021

    code reference: https://msurtsukov.github.io/Neural-ODE/;
    https://github.com/liruilong940607/NeuralODE/blob/master/NeuralODE.py

    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(ODETPP, self).__init__(model_config)

        self.layer_intensity = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_event_types),
            nn.Softplus())

        self.event_model = DNN(inputs_dim=self.hidden_size,
                               hidden_size=[self.hidden_size])

        self.ode_num_sample_per_step = model_config.model_specs['ode_num_sample_per_step']
        self.time_factor = model_config.model_specs['time_factor']

        self.solver = rk4_step_method

        self.layer_neural_ode = NeuralODE(model=self.event_model,
                                          solver=self.solver,
                                          num_sample_times=self.ode_num_sample_per_step)

    def forward(self, time_delta_seqs, type_seqs, **kwargs):
        """Call the model.

        Args:
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.

        """
        # [batch_size, seq_len=max_len-1, hidden_size]
        type_seq_emb = self.layer_type_emb(type_seqs)
        time_delta_seqs_ = time_delta_seqs[..., None]

        total_state_at_event_minus = []
        total_state_at_event_plus = []
        last_state = torch.zeros_like(type_seq_emb[:, 0, :])
        for type_emb, dt in zip(torch.unbind(type_seq_emb, dim=-2),
                                torch.unbind(time_delta_seqs_, dim=-2)):
            dt = dt / self.time_factor
            last_state = self.layer_neural_ode(last_state + type_emb, dt)
            total_state_at_event_minus.append(last_state)
            total_state_at_event_plus.append(last_state + type_emb)

        # [batch_size, seq_len, hidden_size]
        state_ti = torch.stack(total_state_at_event_minus, dim=1)

        # [batch_size, seq_len, hidden_size]
        state_to_evolve = torch.stack(total_state_at_event_plus, dim=1)

        return state_ti, state_to_evolve

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _, type_mask = batch

        state_ti, state_ti_plus = self.forward(time_delta_seqs[:, 1:], type_seqs[:, :-1])

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = state_ti.size()

        # Lambda(t) right before each event time point
        # lambda_at_event - [batch_size, num_times=max_len-1, num_event_types]
        # Here we drop the last event because it has no delta_time label (can not decay)
        lambda_at_event = self.layer_intensity(state_ti)

        # interval_t_sample - [batch_size, num_times=max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seq[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, num_times = max_len - 1, num_mc_sample, hidden_size]
        sample_state_ti = self.compute_states_at_sample_times(state_ti_plus, interval_t_sample)

        # [batch_size, num_times = max_len - 1, num_mc_sample, event_num]
        lambda_t_sample = self.layer_intensity(sample_state_ti)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        loss = - (event_ll - non_event_ll).sum()

        return loss, num_events

    def compute_states_at_sample_times(self, state_ti_plus, sample_dtimes):
        """Compute the states at sampling times.

        Args:
            state_ti_plus (tensor): [batch_size, seq_len, hidden_size], states right after the events.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], delta times in sampling.

        Returns:
            tensor: hiddens states at sampling times.
        """

        # Use broadcasting to compute the decays at all time steps
        # at all sample points
        # h_ts shape (batch_size, seq_len, num_samples, hidden_dim)
        state = self.solver(diff_func=self.event_model,
                            dt=sample_dtimes[..., None],  # [batch_size, seq_len, num_samples, 1]
                            z0=state_ti_plus[..., None, :])  # [batch_size, seq_len, 1, hidden_size]

        return state

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        # forward to the last but one event
        state_ti, state_ti_plus = self.forward(time_delta_seqs, type_seqs, **kwargs)

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = state_ti.size()

        # [batch_size, num_sample_times, num_mc_sample, hidden_size]
        sample_state_ti = self.compute_states_at_sample_times(state_ti_plus, sample_dtimes)

        if compute_last_step_only:
            # [batch_size, 1, num_mc_sample, num_event_types]
            sampled_intensities = self.layer_intensity(sample_state_ti[:, -1:, :, :])
        else:
            # [batch_size, num_sample_times, num_mc_sample, num_event_types]
            sampled_intensities = self.layer_intensity(sample_state_ti)

        return sampled_intensities
