===================================
Model Implementation Details
===================================

Basic structure
===================================

In the model folder, `torch_basemodel` (**/model/torch_model/torch_basemodel.py**) / `tf_basemodel` (**/model/tf_model/tf_basemodel.py**) implements functionalities of computing loglikelihood and sampling procedures that are common
to all the TPP models. In the inherited class, models with specific structures are defined, explained in below sections. 


Computing the loglikelihood of non-pad event sequence
------------------------------------------------------

The loglikelihood computation, following the definition in Equation 8 of `The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process <https://arxiv.org/abs/1612.09328>`_, is shared by all the TPP models.

it takes `time_delta_seqs`, `lambda_at_event`, `lambdas_loss_samples`, `seq_mask`,
                              `lambda_type_mask` as the input and output the loglikelihood items, please see  `torch_basemodel` (**/model/torch_model/torch_basemodel.py**) / `tf_basemodel` (**/model/tf_model/tf_basemodel.py**)
for details.

It is noted that:

1. Sequential prediction: because we performance sequential prediction, i.e., predict next one given previous, we do not consider the last one as it has no labels. To implement the `forward` function, we take input of `time_seqs[:, :-1]`
and `type_seqs[:, :-1]`. For `time_delta_seqs` it is different; please see the next point.



2. Continuous-time evolution: recall the definition in [dataset](./dataset.rst), assume we have a sequence of 4 events and 1 pad event
at the end, i.e.,

.. code-block:: bash

    index:          0,          1,         2,         3，     4
    dtimes:         0，     t_1-t_0,    t_2-t_1,   t_3-t_2,  pad
    types:          e_0,      e_1,        e_2,       e_3，   pad
    non_pad_mask:  True,      True,      True,      True,   False

For the i-th event, i-th dtime denotes the time evolution (e.g., decay in NHP) to the current event and
(i+1)-th dtime denotes the time evolution to the next event. To compute the non-event loglikelihood,
we should consider the time evolution after the event happens. Therefore we should use `type_delta_seqs[:, 1:]` with masks specified in the below step.

3. Masking: suppose we have predictions of 0,1,2,3-th event and their labels are 1,2,3,4-th events
where $4$-th event needed to be masked. So we should set the sequence mask as `True, True, True, False`, i.e., `seq_mask=batch_non_pad_mask[:, 1:]`.
The same logic applies to the attention mask and event type mask.

Therefore the following code is a typical example of calling the loglikelihood computation:


.. code-block:: python

    event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event, # seq_len = max_len - 1
                                                                    lambdas_loss_samples=lambda_t_sample, # seq_len = max_len - 1
                                                                    time_delta_seq=time_delta_seq[:, 1:],
                                                                    seq_mask=batch_non_pad_mask[:, 1:],
                                                                    lambda_type_mask=type_mask[:, 1:])



Computing the integral inside the loglikelihood
-----------------------------------------------


The loglikelihood of the parameters is the sum of the log-intensities of the events that happened, at the times they happened,
minus an integral of the total intensities over the observation interval over [0,T]:

.. math::

    \sum_{t_i}\log \lambda_{k_i}(t_i) - \int_0^T \lambda(t) dt

The first term refers to event loglikelihood and the second term (including the negative sign) refers to the non-event loglikelihood.






Neural Hawkes Process (NHP)
===================================

We implement NHP based on author's official pytorch code `Github:nce-mpp <https://github.com/hongyuanmei/nce-mpp/blob/main/ncempp/models/nhp.py>`_.

1. A continuous-time LSTM is introduced, with the code mainly come from `Github:nce-mpp <https://github.com/hongyuanmei/nce-mpp/blob/main/ncempp/models/nhp.py>`_.
2. A `forward` function in NHP class that recursively update the states: we compute the event embedding, pass to the LSTM cell and then decay afterwards. Noted that for i-th event, we should use (i+1)-th dt for the decay. So we do not consider the last event as it has no decay time.

Attentive Neural Hawkes Process (AttNHP)
========================================


We implement AttNHP based on the authors' official pytorch code `Github:anhp-andtt <https://github.com/yangalan123/anhp-andtt>`_
and similar to NHP, we factorize it into based model and inherited model.

The forward functions is implemented faithfully to that of the author's repo.


Transformer Hawkes Process (THP)
========================================

We implement THP based on a fixed version of pytorch code `Github:anhp-andtt/thp <https://github.com/yangalan123/anhp-andtt/tree/master/thp>`_
and we factorize it into based model and inherited model.


Self-Attentive Hawkes Process (SAHP)
========================================

We implement SAHP based on a fixed version of pytorch code `Github:anhp-andtt/sahp <https://github.com/yangalan123/anhp-andtt/tree/master/sahp>`_
and we factorize it into based model and inherited model.

`SAHP` basically shares very similar structure to that of `THP`.



Recurrent Marked Temporal Point Processes (RMTPP)
====================================================

We implement RMTPP faithfully to the author's paper.


Intensity Free Learning of Temporal Point Process (IntensityFree)
==================================================================

We implement the model based on the author's torch code `Github:ifl-tpp <https://github.com/shchur/ifl-tpp>`_.

A small difference between our implementation and the author's is we ignore the `context_init` (the initial state of the RNN) because in our data setup, we do not need a learnable initial RNN state. This modification generally makes little impact on the learning process.

It is worth noting that the thinning algorithm can not be applied to this model because it is intensity-free. When comparing the performance of the model, we only look at its log-likelihood learning curve.


Fully Neural Network based Model for General Temporal Point Processes (FullyNN)
===============================================================================

We implement the model based on the author's keras code `Github:NeuralNetworkPointProcess <https://github.com/omitakahiro/NeuralNetworkPointProcess>`_.


ODE-based Temporal Point Process (ODETPP)
=========================================

We implement a TPP with Neural ODE state evolution, which is a simplified version of `Neural Spatio-Temporal Point Processes <https://arxiv.org/abs/2011.04583>`_. The ODE implementation uses the code from the `blog <https://msurtsukov.github.io/Neural-ODE/>`_


Attentive Neural Hawkes Network (ANHN)
======================================

We implement the model based on the author's paper: the attentive model without the graph regularizer is named ANHN.
