import argparse

from easy_tpp.config_factory import Config
from easy_tpp.hpo import HyperTuner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='configs/hpo_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='NHP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

    tuner = HyperTuner.build_from_config(config)

    tuner.run()


if __name__ == '__main__':
    main()
