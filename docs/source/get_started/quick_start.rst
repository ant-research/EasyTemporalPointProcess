===========
Quick Start
===========

The smallest current workflow is a YAML experiment plus the registered
``Config`` and ``Runner`` APIs. This example uses the Taxi dataset directly
from `EasyTPP on Hugging Face <https://huggingface.co/easytpp/taxi>`_.


Create a config
===============

Save this as ``taxi.yaml``:

.. code-block:: yaml

    pipeline_config_id: runner_config

    data:
      taxi:
        data_format: json
        train_dir: easytpp/taxi
        valid_dir: easytpp/taxi
        test_dir: easytpp/taxi
        data_specs:
          num_event_types: 10
          pad_token_id: 10
          padding_side: right
          truncation_side: right

    NHP_train:
      base_config:
        stage: train
        backend: torch
        dataset_id: taxi
        runner_id: std_tpp
        model_id: NHP
        base_dir: ./checkpoints/
      trainer_config:
        gpu: -1
        batch_size: 256
        max_epoch: 1
        shuffle: true
        metrics: [acc, rmse]
      model_config:
        hidden_size: 64
        loss_integral_num_sample_per_step: 20
        thinning:
          num_sample: 1
          num_exp: 500
          over_sample_rate: 5
          num_samples_boundary: 5
          dtime_max: 5
          patience_counter: 5
          num_step_gen: 1


Run it
======

Save this as ``train.py`` and run ``python train.py``:

.. code-block:: python

    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner

    config = Config.build_from_yaml_file("taxi.yaml", experiment_id="NHP_train")
    runner = Runner.build_from_config(config)
    runner.run()

The first run downloads the dataset through the Hugging Face ``datasets``
package. To use local pickle or JSON files instead, see `Dataset
<../user_guide/dataset.html>`_.
