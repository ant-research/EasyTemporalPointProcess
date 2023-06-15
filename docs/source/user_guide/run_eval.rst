================================
Evaluate a Model
================================

Step 1: Setup the data and model config files
===============================================

Same as in the training pipeline, firstly we need to initialize two config files:  ``data_config.yaml`` and ``model_config.yaml``.

``data_config.yaml`` is the same as in `Training Pipeline <./run_train_pipeline.html>`_ while we need to update the ``model_config.yaml``
to let the model run the evaluation.


model_config

.. code-block:: yaml

    RMTPP_gen:
        base_config:
            stage: gen
            backend: torch
            dataset_id: retweet
            runner_id: std_tpp
            base_dir: './checkpoints/'
            model_id: RMTPP
        model_config:
            hidden_size: 32
            time_emb_size: 16
            mc_num_sample_per_step: 20
            sharing_param_layer: False
            loss_integral_num_sample_per_step: 20
            dropout: 0.0
            use_ln: False
            seed: 2019
            gpu: 0
            pretrained_model_dir: ./checkpoints/2555_4348724608_230603-155841/models/saved_model
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



A complete example of these files can be seen at `examples/example_config`.


Step 2: Run the evaluation script
=================================

Same as in the training pipeline, we need to initialize a ``ModelRunner`` object to do the evaluation.

The following code is an example, which is a copy from *examples/eval_nhp.py*.


.. code-block:: python

    import argparse

    from easy_tpp.config_factory import RunnerConfig
    from easy_tpp.runner import Runner


    def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                            help='Dir of configuration yaml to train and evaluate the model.')

        parser.add_argument('--experiment_id', type=str, required=False, default='RMTPP_eval',
                            help='Experiment id in the config file.')

        args = parser.parse_args()

        config = RunnerConfig.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

        model_runner = Runner.build_from_config(config)

        model_runner.evaluate()


    if __name__ == '__main__':
        main()




Checkout the output
====================

The evaluation result will be print in the console and saved in the logs whose directory is specified in the
out config file, i.e.:

.. code-block:: bash

    'output_config_dir': './checkpoints/NHP_test_conttime_20221002-13:19:23/NHP_test_output.yaml'
