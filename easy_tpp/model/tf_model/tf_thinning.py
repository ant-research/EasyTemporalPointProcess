import tensorflow as tf
from tensorflow import keras

from easy_tpp.utils.tf_utils import get_shape_list


class EventSampler(keras.Model):
    """
    Event Sequence Sampler based on thinning algorithm

    The algorithm can be found at Algorithm 2 of The Neural Hawkes Process: A Neurally Self-Modulating
    Multivariate Point Process, https://arxiv.org/abs/1612.09328.

    The implementation uses code from https://github.com/yangalan123/anhp-andtt/blob/master/anhp/esm/thinning.py

    """

    def __init__(self, num_sample, num_exp, over_sample_rate, num_samples_boundary, dtime_max, patience_counter):
        super(EventSampler, self).__init__()
        self.num_sample = num_sample  # number of sampled next event times via thinning algorithm,
        # used to compute predictions
        self.num_exp = num_exp  # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.patience_counter = patience_counter

    def compute_intensity_upper_bound(self,
                                      time_seqs,
                                      time_delta_seqs,
                                      type_seqs,
                                      intensity_fn,
                                      compute_last_step_only):
        """

        Args:
            time_seqs: [batch_size, seq_len]
            time_delta_seqs: [batch_size, seq_len]
            type_seqs: [batch_size, seq_len]

        Returns:
            The upper bound of intensity at each event timestamp
            [batch_size, seq_len]

        """

        # [1, 1, num_samples_boundary]
        time_for_bound_sampled = tf.linspace(start=0.0,
                                             stop=1.0,
                                             num=self.num_samples_boundary)[None, None, :]

        # [batch_size, seq_len, num_sample]
        dtime_for_bound_sampled = time_delta_seqs[:, :, None] * time_for_bound_sampled

        # [batch_size, seq_len, num_sample, event_num]
        intensities_for_bound = intensity_fn(time_seqs,
                                             time_delta_seqs,
                                             type_seqs,
                                             dtime_for_bound_sampled,
                                             compute_last_step_only=compute_last_step_only)

        # [batch_size, seq_len]
        bounds = tf.reduce_max(tf.reduce_sum(intensities_for_bound, axis=-1), axis=-1) * self.over_sample_rate

        return bounds

    def sample_exp_distribution(self, sample_rate):
        """

        Args:
            sample_rate: [batch_size, seq_len]
            time_delta_seq: [batch_size, seq_len]

        Returns:
            exp_numbers: [batch_size, seq_len, num_sample, num_exp]

        """

        # can not pass batch_size and seq_len to the random generator as they are dynamics
        # so we reuse rnd for all samples
        batch_size, seq_len = get_shape_list(sample_rate)
        # [batch_size, seq_len, num_exp]
        exp_numbers = tf.random.gamma(shape=[batch_size, seq_len, self.num_exp], alpha=1.0)

        # [batch_size, seq_len, num_exp]
        # exp_numbers = torch.tile(exp_numbers, [1, 1, self.num_sample, 1])

        # [batch_size, seq_len, num_exp]
        # div by sample_rate is equivalent to exp(sample_rate),
        # see https://en.wikipedia.org/wiki/Exponential_distribution
        exp_numbers = exp_numbers / sample_rate[:, :, None]

        return exp_numbers

    def sample_uniform_distribution(self, intensity_upper_bound):
        """

        Returns:
            unif_numbers: [batch_size, seq_len, num_sample, num_exp]

        """
        batch_size, seq_len = get_shape_list(intensity_upper_bound)
        unif_numbers = tf.random.uniform([batch_size, seq_len, self.num_sample, self.num_exp])

        return unif_numbers

    def sample_accept(self, unif_numbers, sample_rate, total_intensities):
        """

        Args:
            unif_numbers: [batch_size, max_len, num_sample, num_exp]
            sample_rate: [batch_size, max_len]
            total_intensities: [batch_size, seq_len, num_sample, num_exp]

        for each parallel draw, find its min criterion： if that < 1.0, the 1st (i.e. smallest) sampled time
        with cri < 1.0 is accepted; if none is accepted, use boundary / maxsampletime for that draw

        Returns:

        """
        # [batch_size, max_len, num_sample, num_exp]
        criterion = unif_numbers * sample_rate[:, :, None, None] / total_intensities

        # [batch_size, max_len, num_sample]
        min_cri_each_draw = tf.reduce_min(criterion, axis=-1)

        # find out unif_numbers * sample_rate < intensity
        # [batch_size, max_len, num_sample]
        who_has_accepted_times = min_cri_each_draw < 1.0

        return criterion, who_has_accepted_times

    def draw_next_time_one_step(self,
                                time_seqs,
                                time_delta_seqs,
                                type_seqs,
                                dtime_boundary,
                                intensity_fn,
                                compute_last_step_only):
        # 1. compute the upper bound of the intensity at each timestamp
        # [batch_size, seq_len]
        intensity_upper_bound = self.compute_intensity_upper_bound(time_seqs,
                                                                   time_delta_seqs,
                                                                   type_seqs,
                                                                   intensity_fn,
                                                                   compute_last_step_only)

        # 2. draw exp distribution with intensity = intensity_upper_bound
        # we apply fast approximation, i.e., re-use exp sample times for computation
        # [batch_size, seq_len, num_exp]
        exp_numbers = self.sample_exp_distribution(intensity_upper_bound)

        # 3. compute intensity at sampled times from exp distribution
        # [batch_size, seq_len, num_exp, event_num]
        intensities_at_sampled_times = intensity_fn(time_seqs,
                                                    time_delta_seqs,
                                                    type_seqs,
                                                    exp_numbers,
                                                    compute_last_step_only=compute_last_step_only)

        # [batch_size, seq_len, num_exp]
        total_intensities = tf.reduce_sum(intensities_at_sampled_times, axis=-1)

        # add one dim of num_sample: re-use the intensity for samples for prediction
        # [batch_size, seq_len, num_sample, num_exp]
        total_intensities = tf.tile(total_intensities[:, :, None, :], [1, 1, self.num_sample, 1])
        # [batch_size, seq_len, num_sample, num_exp]
        exp_numbers = tf.tile(exp_numbers[:, :, None, :], [1, 1, self.num_sample, 1])

        # 4. draw uniform distribution
        # [batch_size, seq_len, num_sample, num_exp]
        unif_numbers = self.sample_uniform_distribution(intensity_upper_bound)

        # 5. find out accepted intensities
        # [batch_size, max_len, num_sample]
        criterion, who_has_accepted_times = self.sample_accept(unif_numbers, intensity_upper_bound,
                                                               total_intensities)

        # 6. find out accepted dtimes
        sampled_dtimes_accepted = tf.identity(exp_numbers)
        # for unaccepted, use boundary/maxsampletime for that draw

        # [batch_size, seq_len, num_sample, num_exp]
        sampled_dtimes_accepted = tf.where(criterion >= 1.0,
                                           tf.ones_like(sampled_dtimes_accepted) * tf.reduce_max(exp_numbers,
                                                                                                 axis=-1,
                                                                                                 keepdims=True) + 1.0,
                                           sampled_dtimes_accepted)

        accepted_times_each_draw = tf.reduce_min(sampled_dtimes_accepted, axis=-1)

        # 7. fill out result
        # [batch_size, seq_len, num_sample]
        dtime_boundary = tf.tile(dtime_boundary[..., None], [1, 1, self.num_sample])

        # [batch_size, seq_len, num_sample]
        res = tf.ones_like(dtime_boundary) * dtime_boundary

        # [batch_size, seq_len, num_sample]
        weights = tf.ones_like(dtime_boundary)
        weights /= tf.reduce_sum(weights, axis=-1, keepdims=True)

        # [batch_size, seq_len, num_sample]
        res = tf.where(who_has_accepted_times,
                       tf.ones_like(res) * accepted_times_each_draw,
                       res)

        who_not_accept = ~who_has_accepted_times

        who_reach_further = exp_numbers[..., -1] > dtime_boundary

        res = tf.where(who_not_accept & who_reach_further,
                       exp_numbers[..., -1],
                       res)

        return res, weights
