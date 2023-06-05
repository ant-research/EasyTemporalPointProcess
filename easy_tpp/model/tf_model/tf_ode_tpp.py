import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from easy_tpp.model.tf_model.tf_baselayer import DNN
from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
from easy_tpp.utils import rk4_step_method
from easy_tpp.utils.tf_utils import get_shape_list

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


def get_neural_ode_layer(
        ode_fn,
        num_samples=10,
        solver=rk4_step_method,
        return_states=False
):
    """Get a black-box neural ode layer parameterized by parameters.

    Args:
        ode_fn: function
            It likes f(solver_function, dt, z_list), and contains the learnable variables.
        num_samples: int
            Number of samples in time interval dt.
        solver: function
            Solver function like f(ode_func, dt, z_0)
        return_states: bool, default False
            Identify whether return whole states or just last state.

    Returns:
        A neural_ode_layer (function) with signature f(z0, dt).
    """

    @tf.custom_gradient
    def neural_ode_layer(
            z0,
            dt,
    ):
        """Calculate z1 by z0 and time gap dt.

        Args:
            z0: Tensor with shape [..., dim]
            dt: Tensor with shape [..., 1 or dim]

        Returns:
            A tensor presents z1, whose shape is the same as z0.
        """
        with tf.name_scope('neural_ode'):
            # Forward activity
            dt_ratio = 1.0 / num_samples
            delta_t = dt * dt_ratio

            z = z0
            z_list = []
            for i in range(num_samples):
                z = solver(ode_fn, delta_t, z)
                z_list.append(z)
            z1 = z

            def grad(a1, variables=None):
                # a1 is grad_z1 == dL/dz1
                if variables is None:
                    variables = []

                def aug_dynamics(tmp_states):
                    """
                    Ode function for states [z_1, a_1, \thetas (many)].

                    Args:
                        tmp_states: list
                            Elements are [z_1, a_1, \thetas (many)].

                    Returns:
                        List contains differentiations of states.
                    """

                    tmp_z = tmp_states[0]
                    tmp_neg_a = -tmp_states[1]
                    # tmp_var_grad = tmp_states[2:]

                    # calculate dz/dt

                    # if tf.__version__ < '2.0':
                    #     # using GradientType to calculate (faster when building graph)
                    #     with tf.GradientTape() as g:
                    #         g.watch([tmp_z, *variables])
                    #         func_eval = ode_fn(tmp_z)
                    #         tmp_ds = g.gradient(func_eval, [tmp_z, *variables], output_gradients=tmp_neg_a)
                    # else:
                    # using tf.gradients to calculate
                    func_eval = ode_fn(tmp_z)
                    tmp_ds = tf.gradients(func_eval, [tmp_z, *variables], grad_ys=tmp_neg_a)

                    neg_adfdz = tmp_ds[0]
                    neg_adfdtheta = [tf.reshape(var, [-1]) for var in tmp_ds[1:]]

                    return [func_eval, neg_adfdz, *neg_adfdtheta]

                # Backward activity
                if tf.__version__ < '2.0':
                    # Compile EAGER graph to static (this will be much faster)
                    import tensorflow.contrib.eager as tfe
                    aug_dynamics = tfe.defun(aug_dynamics)

                # Construct back-state for ode solver
                # reshape variable \theta for batch solving
                init_var_grad = [tf.zeros([np.prod(get_shape_list(var))]) for var in variables]

                if a1 is None:
                    a1 = tf.zeros_like(z1)

                # [z(t_1), a(t_1), \theta]
                states = [z1, a1, *init_var_grad]
                # print('states:', states)
                for i in range(num_samples):
                    states = solver(aug_dynamics, -delta_t, states)

                grad_z0 = states[1]
                grad_t = tf.ones_like(dt)

                if variables is not None:
                    # average the different dt effect on variable \theta
                    grad_theta = [tf.reshape(tf.reduce_mean(var_grad, axis=0), var.shape) for var, var_grad in
                                  zip(variables, states[2:])]
                    return (grad_z0, grad_t), grad_theta
                else:
                    return grad_z0, grad_t

        if return_states:
            return z_list, grad
        else:
            return z1, grad

    return neural_ode_layer


class ODETPP(TfBaseModel):
    def __init__(self, model_config):
        super(ODETPP, self).__init__(model_config)
        self.ode_num_sample_per_step = model_config.specs['ode_num_sample_per_step']
        self.time_factor = model_config.specs['time_factor']
        self.seq_len = model_config.max_len

    def build_graph(self):
        """Build up the network
        """
        with tf.variable_scope('ODETPP'):
            # have to specify the max len of the input to avoid a variable length of tensor.
            # for looping over the variable length of tensor, custom gradient can not properly work
            # in the scan (while_loop)
            # tf.GradientTape.gradients() does not support graph control flow operations
            # like tf.cond or tf.while at this time

            # Input placeholder
            # shape - (batch_size, max_len)
            # max_len - sequence length including time zero padding
            self.time_delta_seqs = tf.placeholder(tf.float32, shape=[None, self.seq_len])

            # shape - (batch_size, max_len)
            self.time_seqs = tf.placeholder(tf.float32, shape=[None, self.seq_len])

            # shape - (batch_size, max_len)
            self.type_seqs = tf.placeholder(tf.int32, shape=[None, self.seq_len])

            # shape - (batch_size, max_len)
            self.batch_non_pad_mask = tf.placeholder(tf.int32, shape=[None, self.seq_len])

            # shape - (batch_size, max_len, max_len)
            self.attention_mask = tf.placeholder(tf.int32, shape=[None, None, None])

            # Event type one-hot code
            # shape - (batch_size, max_len, num_event_types)
            self.type_mask = tf.placeholder(tf.float32, shape=[None, None, None])

            self.layer_intensity = layers.Dense(self.num_event_types, activation=tf.nn.softplus)

            self.event_model = DNN(hidden_size=self.hidden_size)

            self.solver = rk4_step_method

            self.layer_neural_ode = get_neural_ode_layer(ode_fn=self.event_model,
                                                         solver=self.solver,
                                                         num_samples=self.ode_num_sample_per_step)

            self.loss, self.num_event = self.loglike_loss()

            self.is_training = tf.placeholder(tf.bool)

            # Make predictions
            if self.event_sampler and self.gen_config.num_step_gen == 1:
                self.dtime_predict_one_step, self.type_predict_one_step = \
                    self.predict_one_step_at_every_event(self.time_seqs,
                                                         self.time_delta_seqs,
                                                         self.type_seqs)

            if self.event_sampler and self.gen_config.num_step_gen > 1:
                # make generations
                self.dtime_generation, self.type_generation = \
                    self.predict_multi_step_since_last_event(self.time_seqs,
                                                             self.time_delta_seqs,
                                                             self.type_seqs,
                                                             num_step=self.gen_config.num_step_gen)

    def forward(self, time_delta_seqs, type_seqs, **kwargs):
        """Call the model.

        Args:
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.

        """
        # [batch_size, seq_len=max_len, hidden_size]
        type_seq_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len=max_len, 1]
        time_delta_seqs_ = time_delta_seqs[..., None]

        total_state_at_event_minus = []
        total_state_at_event_plus = []
        last_state = tf.zeros_like(type_seq_emb[:, 0, :])
        for type_emb, dt in zip(tf.unstack(type_seq_emb, axis=-2),
                                tf.unstack(time_delta_seqs_, axis=-2)):
            # the bp may break for tf 1.13 when dt is large
            # after testing, we put a time factor here to avoid the failure of bp
            # it is not needed for tf 2.0.
            dt = dt / self.time_factor
            last_state = self.layer_neural_ode(last_state + type_emb, dt)
            total_state_at_event_minus.append(last_state)
            total_state_at_event_plus.append(last_state + type_emb)

        # [batch_size, seq_len, hidden_size]
        state_ti = tf.stack(total_state_at_event_minus, axis=1)

        # [batch_size, seq_len, hidden_size]
        state_to_evolve = tf.stack(total_state_at_event_plus, axis=1)

        return state_ti, state_to_evolve

    def loglike_loss(self):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """

        state_ti, state_ti_plus = self.forward(self.time_delta_seqs[:, 1:], self.type_seqs[:, :-1])

        # Lambda(t) right before each event time point
        # lambda_at_event - [batch_size, num_times=max_len-1, num_event_types]
        # Here we drop the last event because it has no delta_time label (can not decay)
        lambda_at_event = self.layer_intensity(state_ti)

        # interval_t_sample - [batch_size, num_times=max_len-1, num_mc_sample]
        # for every batch and every event point => do a sampling (num_mc_sampling)
        # the first dtime is zero, so we use time_delta_seq[:, 1:]
        interval_t_sample = self.make_dtime_loss_samples(self.time_delta_seqs[:, 1:])

        # [batch_size, num_times = max_len - 1, num_mc_sample, hidden_size]
        sample_state_ti = self.compute_states_at_sample_times(state_ti_plus, interval_t_sample)

        # [batch_size, num_times = max_len - 1, num_mc_sample, event_num]
        lambda_t_sample = self.layer_intensity(sample_state_ti)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=self.time_delta_seqs[:, 1:],
                                                                        seq_mask=self.batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=self.type_mask[:, 1:])

        loss = - tf.reduce_sum(event_ll - non_event_ll)

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

        state_ti, state_ti_plus = self.forward(time_delta_seqs, type_seqs, **kwargs)

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = get_shape_list(state_ti)

        if compute_last_step_only:
            interval_t_sample = sample_dtimes[:, -1:, :]
        else:
            # interval_t_sample - [batch_size, num_times, num_mc_sample, 1]
            interval_t_sample = sample_dtimes
            # Use broadcasting to compute the decays at all time steps
            # at all sample points

        # [batch_size, num_sample_times / 1, num_mc_sample, hidden_size]
        sample_state_ti = self.compute_states_at_sample_times(state_ti_plus, interval_t_sample)

        # [batch_size, num_sample_times / 1, num_mc_sample, num_event_types]
        sampled_intensities = self.layer_intensity(sample_state_ti)

        return sampled_intensities
