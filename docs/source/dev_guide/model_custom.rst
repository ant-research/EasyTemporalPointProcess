==================
Customize a Model
==================


Here we introduce how to customize a TPP model with the support of ``EasyTPP``.



Create a new TPP Model Class
=============================

Assume we are building a PyTorch model. We need to initialize the model by inheriting class `EasyTPP.model.torch_model.TorchBaseModel <../ref/models.html>`_.

.. code-block:: python

    from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

    # Custom Torch TPP implementations need to
    # inherit from the TorchBaseModel interface
    class NewModel(TorchBaseModel):
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


If we are building a Tensorflow model, we start with the following code

.. code-block:: python

    from easy_tpp.model.torch_model.tf_basemodel import TfBaseModel

    # Custom Tf TPP implementations need to
    # inherit from the TorchBaseModel interface
    class NewModel(TfBaseModel):
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
