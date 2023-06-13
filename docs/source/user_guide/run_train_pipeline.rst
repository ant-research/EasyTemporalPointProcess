============================================
Training a Model & Configuration Explanation
============================================

This tutorial shows how one can use ``EasyTPP`` to train the implemented models.

In principle, firstly we need to initialize a config yaml file, containing all the input configuration to guide the training and eval process. The overall structure of a config file is shown as below:

.. code-block:: yaml

    pipeline_config_id: ..  # name of the config for guiding the pipeline

    data:
        [Dataset ID]:  # name of the dataset, e.g, taxi
            ....

    [EXPERIMENT ID]:   # name of the experiment to run
        base_config:
          ....
        model_config:
           ...


After the config file is setup, we can run the script, by specifying the `config directory` and `experiment id`, to start the pipeline. We currently provide a preset script at `examples/train_nhp.py`.


Step 1: Setup the config file containing data and model configs
================================================================


To be specific, one needs to define the following entries in the config file:

- **pipeline_config_id**: registered name of EasyTPP.Config objects, such as `runner_config` or `hpo_runner_config`. By reading this, the corresponding configuration class will be loaded for constructing the pipeline.

.. code-block:: yaml

    pipeline_config_id: runner_config


- **data**:  dataset specifics. One can put multiple dataset specifics in the config file, but only one will be used in one experiment.

    - *[DATASET ID]*: name of the dataset, e.g., taxi.
    - *train_dir, valid_dir, test_dir*: directory of the datafile. For the moment we only accept pkl file (please see `Dataset <./dataset.html>`_ for details)
    - *data_spec*: define the event type information.

.. code-block:: yaml

    data:
      taxi:
        data_format: pkl
        train_dir: ../data/taxi/train.pkl
        valid_dir: ../data/taxi/dev.pkl
        test_dir: ../data/taxi/test.pkl
        data_spec：
            num_event_types: 7  # num of types excluding pad events.
            pad_token_id: 6    # event type index for pad events
            padding_side: right   # pad at the right end of the sequence
            truncation_side: right   # truncate at the right end of the sequence
            max_len: 100            # max sequence length used as model input

- **[EXPERIMENT ID]**: name of the experiment to run in the pipeline. It contains two blocks of configs:

*base_config* contains the pipeline framework related specifications.

.. code-block:: yaml

    base_config:
        stage: train       # train, eval and generate
        backend: tensorflow   # tensorflow and torch
        dataset_id: conttime   # name of the dataset
        runner_id: std_tpp     # registered name of the pipeline runner
        model_id: RMTPP # model name  # registered name of the implemented model
        base_dir: './checkpoints/'   # base dir to save the logs and models.



*model_config* contains the model related specifications.


.. code-block:: yaml

      model_config:
        hidden_size: 32
        time_emb_size: 16
        num_layers: 2
        num_heads: 2
        mc_num_sample_per_step: 20
        sharing_param_layer: False
        loss_integral_num_sample_per_step: 20
        dropout: 0.0
        use_ln: False
        seed: 2019
        gpu: 0
        trainer:   # trainer arguments
          batch_size: 256
          max_epoch: 10
          shuffle: False
          optimizer: adam
          learning_rate: 1.e-3
          valid_freq: 1
          use_tfb: False
          metrics: ['acc', 'rmse']
        thinning_params:   # thinning algorithm for event sampling
          num_seq: 10
          num_sample: 1
          num_exp: 500 # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
          look_ahead_time: 10
          patience_counter: 5 # the maximum iteration used in adaptive thinning
          over_sample_rate: 5
          num_samples_boundary: 5
          dtime_max: 5




A complete example of these files can be seen at *examples/example_config*.


Step 2: Run the training script
===============================================

To run the training process, we simply need to call two functions:

1. ``Config``: it reads the directory of the configs specified in Step 1 and do some processing to form a complete configuration.
2. ``Runner``: it reads the configuration and setups the whole pipeline for training, evaluation and generation.


The following code is an example, which is a copy from *examples/train_nhp.py*.


.. code-block:: python

    import argparse
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner


    def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                            help='Dir of configuration yaml to train and evaluate the model.')

        parser.add_argument('--experiment_id', type=str, required=False, default='RMTPP_train',
                            help='Experiment id in the config file.')

        args = parser.parse_args()

        config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

        model_runner = Runner.build_from_config(config)

        model_runner.run()


    if __name__ == '__main__':
        main()





Checkout the output
========================


During training, the log, the best model based on valid set performance, the complete configuration file are all saved. The directory of the saved files is specified in 'base' of ``model_config.yaml``, i.e.,



In the `./checkpoints/` folder, one find the correct subfolder by concatenating the 'experiment_id' and running timestamps. Inside that subfolder, there is a complete configuration file, e.g., ``NHP_train_output.yaml`` that records all the information used in the pipeline. The

.. code-block:: yaml

    data_config:
      train_dir: ../data/conttime/train.pkl
      valid_dir: ../data/conttime/dev.pkl
      test_dir: ../data/conttime/test.pkl
      specs:
        num_event_types_pad: 6
        num_event_types: 5
        event_pad_index: 5
      data_format: pkl
    base_config:
      stage: train
      backend: tensorflow
      dataset_id: conttime
      runner_id: std_tpp
      model_id: RMTPP
      base_dir: ./checkpoints/
      exp_id: RMTPP_train
      log_folder: ./checkpoints/98888_4299965824_221205-153425
      saved_model_dir: ./checkpoints/98888_4299965824_221205-153425/models/saved_model
      saved_log_dir: ./checkpoints/98888_4299965824_221205-153425/log
      output_config_dir: ./checkpoints/98888_4299965824_221205-153425/RMTPP_train_output.yaml
    model_config:
      hidden_size: 32
      time_emb_size: 16
      num_layers: 2
      num_heads: 2
      mc_num_sample_per_step: 20
      sharing_param_layer: false
      loss_integral_num_sample_per_step: 20
      dropout: 0.0
      use_ln: false
      seed: 2019
      gpu: 0
      thinning_params:
        num_seq: 10
        num_sample: 1
        num_exp: 500
        look_ahead_time: 10
        patience_counter: 5
        over_sample_rate: 5
        num_samples_boundary: 5
        dtime_max: 5
        num_step_gen: 1
      trainer:
        batch_size: 256
        max_epoch: 10
        shuffle: false
        optimizer: adam
        learning_rate: 0.001
        valid_freq: 1
        use_tfb: false
        metrics:
        - acc
        - rmse
        seq_pad_end: true
      is_training: true
      num_event_types_pad: 6
      num_event_types: 5
      event_pad_index: 5
      model_id: RMTPP



If we set ``use_tfb`` to ``true``, it means we can launch the tensorboard to track the training process, one
can see `Running Tensorboard <../advanced/tensorboard.html>`_ for details.