import torch
import torch.nn as nn
from easy_tpp.utils import logger


class EventSampler(nn.Module):
    """Event Sequence Sampler based on thinning algorithm, which corresponds to Algorithm 2 of
    The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process,
    https://arxiv.org/abs/1612.09328.

    The implementation uses code from https://github.com/yangalan123/anhp-andtt/blob/master/anhp/esm/thinning.py.
    """

    def __init__(self, num_sample, num_exp, over_sample_rate, num_samples_boundary, dtime_max, patience_counter,
                 device):
        """Initialize the event sampler.

        Args:
            num_sample (int): number of sampled next event times via thinning algo for computing predictions.
            num_exp (int): number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
            over_sample_rate (float): multiplier for the intensity up bound.
            num_samples_boundary (int): number of sampled event times to compute the boundary of the intensity.
            dtime_max (float): max value of delta times in sampling
            patience_counter (int): the maximum iteration used in adaptive thinning.
            device (torch.device): torch device index to select.
        """
        super(EventSampler, self).__init__()
        self.num_sample = num_sample
        self.num_exp = num_exp
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.patience_counter = patience_counter
        self.device = device

    def compute_intensity_upper_bound(self, time_seq, time_delta_seq, event_seq, intensity_fn,
                                      compute_last_step_only):
        # logger.critical(f'time_seq: {time_seq}')
        # logger.critical(f'time_delta_seq: {time_delta_seq}')
        # logger.critical(f'event_seq: {event_seq}')
        # logger.critical(f'intensity_fn: {intensity_fn}')
        # logger.critical(f'compute_last_step_only: {compute_last_step_only}')
        """Compute the upper bound of intensity at each event timestamp.

        Args:
            time_seq (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            intensity_fn (fn): a function that computes the intensity.
            compute_last_step_only (bool): wheter to compute the last time step pnly.

        Returns:
            tensor: [batch_size, seq_len]
        """
        batch_size, seq_len = time_seq.size()

        # [1, 1, num_samples_boundary]
        time_for_bound_sampled = torch.linspace(start=0.0,
                                                end=1.0,
                                                steps=self.num_samples_boundary,
                                                device=self.device)[None, None, :]

        # [batch_size, seq_len, num_samples_boundary]
        dtime_for_bound_sampled = time_delta_seq[:, :, None] * time_for_bound_sampled

        # [batch_size, seq_len, num_samples_boundary, event_num]
        intensities_for_bound = intensity_fn(time_seq,
                                             time_delta_seq,
                                             event_seq,
                                             dtime_for_bound_sampled,
                                             max_steps=seq_len,
                                             compute_last_step_only=compute_last_step_only)

        # [batch_size, seq_len]
        bounds = intensities_for_bound.sum(dim=-1).max(dim=-1)[0] * self.over_sample_rate

        return bounds

    def sample_exp_distribution(self, sample_rate):
        """Sample an exponential distribution.

        Args:
            sample_rate (tensor): [batch_size, seq_len], intensity rate.

        Returns:
            tensor: [batch_size, seq_len, num_exp], exp numbers at each event timestamp.
        """

        batch_size, seq_len = sample_rate.size()

        # For fast approximation, we reuse the rnd for all samples
        # [batch_size, seq_len, num_exp]
        exp_numbers = torch.empty(size=[batch_size, seq_len, self.num_exp],
                                  dtype=torch.float32,
                                  device=self.device)

        # [batch_size, seq_len, num_exp]
        # exp_numbers.exponential_(1.0)
        exp_numbers.exponential_(1.0)

        # [batch_size, seq_len, num_exp]
        # exp_numbers = torch.tile(exp_numbers, [1, 1, self.num_sample, 1])

        # [batch_size, seq_len, num_exp]
        # div by sample_rate is equivalent to exp(sample_rate),
        # see https://en.wikipedia.org/wiki/Exponential_distribution
        exp_numbers = exp_numbers / sample_rate[:, :, None]

        return exp_numbers

    def sample_uniform_distribution(self, intensity_upper_bound):
        """Sample an uniform distribution

        Args:
            intensity_upper_bound (tensor): upper bound intensity computed in the previous step.

        Returns:
            tensor: [batch_size, seq_len, num_sample, num_exp]
        """
        batch_size, seq_len = intensity_upper_bound.size()

        unif_numbers = torch.empty(size=[batch_size, seq_len, self.num_sample, self.num_exp],
                                   dtype=torch.float32,
                                   device=self.device)
        unif_numbers.uniform_(0.0, 1.0)

        return unif_numbers

    def sample_accept(self, unif_numbers, sample_rate, total_intensities, exp_numbers):
        """Do the sample-accept process.

        For the accumulated exp (delta) samples drawn for each event timestamp, find (from left to right) the first
        that makes the criterion < 1 and accept it as the sampled next-event time. If all exp samples are rejected 
        (criterion >= 1), then we set the sampled next-event time dtime_max.

        Args:
            unif_numbers (tensor): [batch_size, max_len, num_sample, num_exp], sampled uniform random number.
            sample_rate (tensor): [batch_size, max_len], sample rate (intensity).
            total_intensities (tensor): [batch_size, seq_len, num_sample, num_exp]
            exp_numbers (tensor): [batch_size, seq_len, num_sample, num_exp]: sampled exp numbers (delta in Algorithm 2).

        Returns:
            result (tensor): [batch_size, seq_len, num_sample], sampled next-event times.
        """

        # [batch_size, max_len, num_sample, num_exp]
        criterion = unif_numbers * sample_rate[:, :, None, None] / total_intensities
        
        # [batch_size, max_len, num_sample, num_exp]
        masked_crit_less_than_1 = torch.where(criterion<1,1,0)
        
        # [batch_size, max_len, num_sample]
        non_accepted_filter = (1-masked_crit_less_than_1).all(dim=3)
        
        # [batch_size, max_len, num_sample]
        first_accepted_indexer = masked_crit_less_than_1.argmax(dim=3)
        
        # [batch_size, max_len, num_sample,1]
        # indexer must be unsqueezed to 4D to match the number of dimensions of exp_numbers
        result_non_accepted_unfiltered = torch.gather(exp_numbers, 3, first_accepted_indexer.unsqueeze(3))
        
        # [batch_size, max_len, num_sample,1]
        result = torch.where(non_accepted_filter.unsqueeze(3), torch.tensor(self.dtime_max), result_non_accepted_unfiltered)
        
        # [batch_size, max_len, num_sample]
        result = result.squeeze(dim=-1)
        
        return result

    def draw_next_time_one_step(self, time_seq, time_delta_seq, event_seq, dtime_boundary,
                                intensity_fn, compute_last_step_only=False):
        """Compute next event time based on Thinning algorithm.

        Args:
            time_seq (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            dtime_boundary (tensor): [batch_size, seq_len], dtime upper bound.
            intensity_fn (fn): a function to compute the intensity.
            compute_last_step_only (bool, optional): whether to compute last event timestep only. Defaults to False.

        Returns:
            tuple: next event time prediction and weight.
        """
        # 1. compute the upper bound of the intensity at each timestamp
        # the last event has no label (no next event), so we drop it
        # [batch_size, seq_len=max_len - 1]
        intensity_upper_bound = self.compute_intensity_upper_bound(time_seq,
                                                                   time_delta_seq,
                                                                   event_seq,
                                                                   intensity_fn,
                                                                   compute_last_step_only)

        # 2. draw exp distribution with intensity = intensity_upper_bound
        # we apply fast approximation, i.e., re-use exp sample times for computation
        # [batch_size, seq_len, num_exp]
        exp_numbers = self.sample_exp_distribution(intensity_upper_bound)
        exp_numbers = torch.cumsum(exp_numbers, dim=-1)
        
        # 3. compute intensity at sampled times from exp distribution
        # [batch_size, seq_len, num_exp, event_num]
        intensities_at_sampled_times = intensity_fn(time_seq,
                                                    time_delta_seq,
                                                    event_seq,
                                                    exp_numbers,
                                                    max_steps=time_seq.size(1),
                                                    compute_last_step_only=compute_last_step_only)

        # [batch_size, seq_len, num_exp]
        total_intensities = intensities_at_sampled_times.sum(dim=-1)

        # add one dim of num_sample: re-use the intensity for samples for prediction
        # [batch_size, seq_len, num_sample, num_exp]
        total_intensities = torch.tile(total_intensities[:, :, None, :], [1, 1, self.num_sample, 1])
        
        # [batch_size, seq_len, num_sample, num_exp]
        exp_numbers = torch.tile(exp_numbers[:, :, None, :], [1, 1, self.num_sample, 1])
        
        # 4. draw uniform distribution
        # [batch_size, seq_len, num_sample, num_exp]
        unif_numbers = self.sample_uniform_distribution(intensity_upper_bound)

        # 5. find out accepted intensities
        # [batch_size, seq_len, num_sample]
        res = self.sample_accept(unif_numbers, intensity_upper_bound, total_intensities, exp_numbers)

        # [batch_size, seq_len, num_sample]
        weights = torch.ones_like(res)/res.shape[2]
        
        # add a upper bound here in case it explodes, e.g., in ODE models
        return res.clamp(max=1e5), weights
