==============================================
Training a Model and Configuring the Pipeline
==============================================

An EasyTPP experiment file contains one ``data`` mapping and one or more
named experiments. ``pipeline_config_id: runner_config`` selects
``RunnerConfig``; the experiment's ``runner_id: std_tpp`` selects the standard
torch TPP runner.


Minimal training configuration
==============================

This complete example trains NHP on the hosted Taxi dataset:

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
          rescale_time: false

    NHP_train:
      base_config:
        stage: train
        backend: torch
        dataset_id: taxi
        runner_id: std_tpp
        model_id: NHP
        base_dir: ./checkpoints/
      trainer_config:
        seed: 2019
        gpu: -1
        batch_size: 256
        max_epoch: 10
        shuffle: true
        optimizer: adam
        learning_rate: 0.001
        valid_freq: 1
        use_tfb: false
        metrics: [acc, rmse]
      model_config:
        hidden_size: 64
        time_emb_size: 16
        num_layers: 2
        num_heads: 2
        use_mc_samples: true
        sharing_param_layer: false
        loss_integral_num_sample_per_step: 20
        dropout_rate: 0.0
        use_ln: false
        thinning:
          num_sample: 1
          num_exp: 500
          over_sample_rate: 5
          num_samples_boundary: 5
          dtime_max: 5
          patience_counter: 5
          num_step_gen: 1

``data_format`` is either ``json`` or ``pkl``. See `Dataset
<./dataset.html>`_ for local layouts, hosted datasets, and the optional
``rescale_time`` / ``time_scale`` settings.

The thinning keys above match ``ThinningConfig`` and the current generation
path. ``num_sample`` controls the number of next-time samples, ``num_exp`` the
number of exponential proposals, ``over_sample_rate`` the upper-bound
multiplier, ``num_samples_boundary`` the number of intensity probes used for
the bound, ``dtime_max`` the fallback interval, and ``num_step_gen`` the
generation horizon. ``patience_counter`` remains part of the public config
and is passed to the sampler.

Model-specific options belong in ``model_config.model_specs``. The shared
options shown above are parsed by ``ModelConfig``; training options such as
``gpu`` and ``seed`` belong in ``trainer_config``.


Run the experiment
==================

The current API constructs the registered config with ``Config`` and the
registered runner with ``Runner``:

.. code-block:: python

    import argparse

    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner


    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs/experiment_config.yaml")
    parser.add_argument("--experiment_id", default="NHP_train")
    args = parser.parse_args()

    config = Config.build_from_yaml_file(
        args.config_dir,
        experiment_id=args.experiment_id,
    )
    runner = Runner.build_from_config(config)
    runner.run()

This is the same entry-point pattern used by `examples/train_nhp.py
<https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/train_nhp.py>`_.
The run directory below ``base_dir`` contains logs, the saved model, and the
resolved output configuration. Set ``use_tfb: true`` to produce TensorBoard
logs; see `Running TensorBoard <../advanced/tensorboard.html>`_.
