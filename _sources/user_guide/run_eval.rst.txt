================
Evaluate a Model
================

Evaluation uses the same config and runner API as training. Keep the dataset
entry and architecture settings used for training, set ``stage`` to ``eval``,
and point ``model_config.pretrained_model_dir`` at the saved model.


Evaluation configuration
========================

The following experiment can live beside the ``data.taxi`` entry shown in
`Training a Model <./run_train_pipeline.html>`_:

.. code-block:: yaml

    NHP_eval:
      base_config:
        stage: eval
        backend: torch
        dataset_id: taxi
        runner_id: std_tpp
        model_id: NHP
        base_dir: ./checkpoints/
      trainer_config:
        seed: 2019
        gpu: -1
        batch_size: 256
        max_epoch: 1
        shuffle: false
        metrics: [acc, rmse]
      model_config:
        hidden_size: 64
        time_emb_size: 16
        num_layers: 2
        num_heads: 2
        use_mc_samples: true
        loss_integral_num_sample_per_step: 20
        dropout_rate: 0.0
        use_ln: false
        pretrained_model_dir: ./checkpoints/<run-id>/models/saved_model
        thinning:
          num_sample: 1
          num_exp: 500
          over_sample_rate: 5
          num_samples_boundary: 5
          dtime_max: 5
          patience_counter: 5
          num_step_gen: 1

The architecture must match the checkpoint. Thinning is used for next-event
time prediction by intensity-based models. The supported thinning keys are
``num_sample``, ``num_exp``, ``over_sample_rate``,
``num_samples_boundary``, ``dtime_max``, ``patience_counter``, and
``num_step_gen``.


Run evaluation
==============

Use ``Config``, not a separate evaluation config class:

.. code-block:: python

    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner

    config = Config.build_from_yaml_file(
        "configs/experiment_config.yaml",
        experiment_id="NHP_eval",
    )
    runner = Runner.build_from_config(config)
    runner.run()

Metrics are printed and written to the log below ``base_dir``. When dataset
time rescaling is active, predicted and target times are restored to their
original units before RMSE is calculated.
