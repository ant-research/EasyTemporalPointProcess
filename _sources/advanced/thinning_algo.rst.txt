==============================================
Thinning Algorithm for Sampling Event Sequence
==============================================

In ``EasyTPP`` we use ``Thinning algorithm`` depicted in Algorithm 2
in `The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process <https://arxiv.org/abs/1612.09328>`_
for event sampling.

The implementation of the algorithm
====================================


We implement the algorithm both in PyTorch and Tensorflow, as seen in *./model/torch_thinning.py* and
*./model/tf_thinning.py*, which basically follow the same procedure.

The corresponding code is in function ``draw_next_time_one_step``, which consists of the following steps:

1. Compute the upper bound of the intensity at each event timestamp in function ``compute_intensity_upper_bound``, where we sample some timestamps inside event intervals and output a upper bound intensity matrix [batch_size, seq_len]， denoting the upper bound of prediced intensity (for next time interval) for each sequence at each timestamp.
2. Sample the exponential distribution with the intensity computed in Step 1 in function ``sample_exp_distribution``, where we simply divide the standard exponential number with the intensity, which is equivalent to sampling with exp(sample_rate), according to `the properties of Exponential Distribution <https://en.wikipedia.org/wiki/Exponential_distribution>`_. The exponential random variables have size [batch_size, seq_len, num_sample, num_exp], where num_sample refers to the number of event times sampled in every interval and num_exp refers to number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm.
3. Compute the intensities at the sample times proposed in Step 2， with final size `[batch_size, seq_len, num_sample, num_exp]`.
4. Sample the standard uniform distribution with size `[batch_size, seq_len, num_sample, num_exp]`.
5. Perform the acceptance sampling with certain probability in function ``sample_accept``.
6. The earliest sampling dtimes are accepted. For unaccepted sampling dtimes, use boundary/maxsampletime for that draw.
7. The final predicted dtimes has size `[batch_size, seq_len, num_sample]`, which refers to the sampling dtimes for each sequence at each timestamp, along with an equal weight vector.
8. The product of the predicted dtimes and the weight is the final predicted dtimes, with size `[batch_size, seq_len]`.


.. image:: ../../images/thinning_algo.jpg
    :alt: thinning_algo



One-step prediction
====================================
By default, once given the parameters of thinning algo (defining a ``thinning`` config as part of ``model_config``), we perform the one-step prediction in model evaluation, i.e., predict the next event given the prefix. The implementation is in function ``prediction_event_one_step`` in BaseModel (i.e., TorchBaseModel or TfBaseModel).


Multi-step prediction
====================================
The recursive multi-step prediction is activated by setting `num_step_gen` to a number bigger than 1 in the ``thinning`` config.

Be noted that, we generate the multi-step events after the last non-pad event of each sequence. The implementation is in function `predict_multi_step_since_last_event` in BaseModel (i.e., TorchBaseModel or TfBaseModel).


.. code-block:: yaml

    thinning:
      num_seq: 10
      num_sample: 1
      num_exp: 500 # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
      look_ahead_time: 10
      patience_counter: 5 # the maximum iteration used in adaptive thinning
      over_sample_rate: 5
      num_samples_boundary: 5
      dtime_max: 5
      num_step_gen: 5    # by default it is single step, i.e., 1