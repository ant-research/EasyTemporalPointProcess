===================================
Launching the Tensorboard
===================================


Here we present how to launch the tensorboard within the  ``EasyTPP`` framework.

Step 1: Activate the usage of tensorboard in Config file
========================================================


As shown in `Training Pipeline <../get_started/run_train_pipeline.html>`_, we need to firstly initialize the 'model_config.yaml' file to setup the running config before training or evaluating the model.

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
        shuffle: False
        optimizer: adam
        learning_rate: 1.e-3
        valid_freq: 1
        use_tfb: True  # Activate the tensorboard
        metrics: [ 'acc', 'rmse' ]
        seed: 2019
        gpu: -1
      model_config:
        hidden_size: 64
        loss_integral_num_sample_per_step: 20
    #    pretrained_model_dir: ./checkpoints/75518_4377527680_230530-132355/models/saved_model
        thinning:
          num_seq: 10
          num_sample: 1
          num_exp: 500 # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
          look_ahead_time: 10
          patience_counter: 5 # the maximum iteration used in adaptive thinning
          over_sample_rate: 5
          num_samples_boundary: 5
          dtime_max: 5
          num_step_gen: 1



Step 2: Launching the tensorboard
========================================================


We simply go to the output file of the training runner (its directory is specified in `base_dir` of ``base_config``), find out the tensorboard file address and launch it.

A complete example of using tensorboard can be seen at *examples/run_tensorboard.py*.


.. code-block:: python

    import os

    def main():
        # one can find this dir in the config out file
        log_dir = './checkpoints/NHP_train_taxi_20220527-20:18:30/tfb_train'
        os.system('tensorboard --logdir={}'.format(log_dir))
        return


    if __name__ == '__main__':
        main()