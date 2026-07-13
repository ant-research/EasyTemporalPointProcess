==================
Customize a Model
==================


Here we introduce how to customize a TPP model with the support of ``EasyTPP``.



Create a new TPP Model Class
=============================

New models inherit from
`easy_tpp.model.BaseModel <../ref/models.html>`_. ``TorchBaseModel`` remains an
alias for backward compatibility.

.. code-block:: python

    from easy_tpp.model.basemodel import BaseModel

    # Custom Torch TPP implementations need to
    # inherit from the BaseModel interface
    class NewModel(BaseModel):
        def __init__(self, model_config):
            super(NewModel, self).__init__(model_config)

        # Forward along the sequence, output the states / intensities at the event times
        def forward(self, batch):
            ...
            return states

        # Compute the loglikelihood loss
        def loglike_loss(self, batch):
            ....
            return loglike

        # Compute the intensities at given sampling times
        # Used in the Thinning sampler
        def compute_intensities_at_sample_times(self, batch, sample_times, **kwargs):
            ...
            return intensities


Rewrite Relevant Methods
==============================

There are three important functions needed to be implemented:

- `forward`: the input is the batch data and the output is states at each step.
- `loglike_loss`: it computes the loglikihood loss given the batch data.
- `compute_intensities_at_sample_times`: it computes the intensities at each sampling steps.
