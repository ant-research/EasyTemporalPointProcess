===================================
Launching the Tensorboard
===================================


Here we present how to launch the tensorboard within the  ``EasyTPP`` framework.

Step 1: Activate the usage of tensorboard in Config file
========================================================


As shown in `Training Pipeline <../user_guide/run_train_pipeline.html>`_, enable
TensorBoard in the experiment's ``trainer_config`` before training.

In the ``model config`` (`modeling` attribute of the config), one needs to set ``use_tfb`` to ``True`` in `trainer`. Then before the running process, summary writers tracking the performance on training and valid sets are both initialized.

.. code-block:: yaml

    NHP_train:
      base_config:
        stage: train
        backend: torch
        dataset_id: taxi
        runner_id: std_tpp
        model_id: NHP # model name
        base_dir: './checkpoints/'
      trainer_config:
        batch_size: 256
        max_epoch: 200
        shuffle: false
        optimizer: adam
        learning_rate: 1.e-3
        valid_freq: 1
        use_tfb: true  # activate TensorBoard
        metrics: [ 'acc', 'rmse' ]
        seed: 2019
        gpu: -1
      model_config:
        hidden_size: 64
        loss_integral_num_sample_per_step: 20
        thinning:
          num_sample: 1
          num_exp: 500 # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
          patience_counter: 5 # the maximum iteration used in adaptive thinning
          over_sample_rate: 5
          num_samples_boundary: 5
          dtime_max: 5
          num_step_gen: 1



Step 2: Launching the tensorboard
========================================================


Find ``tfb_train`` in the timestamped run directory below ``base_dir`` and
launch TensorBoard from the shell:

.. code-block:: bash

    tensorboard --logdir ./checkpoints/<run-id>/tfb_train
