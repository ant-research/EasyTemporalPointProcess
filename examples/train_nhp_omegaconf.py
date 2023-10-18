from omegaconf import OmegaConf

from easy_tpp.config_factory import ModelConfig
from easy_tpp.model.torch_model.torch_nhp import NHP


def main():
    config_omegaconf = OmegaConf.load('configs/experiment_config.yaml')

    model_config_dict = config_omegaconf.get('NHP_train').get('model_config')
    model_config_dict['num_event_types'] = 10
    model_config_dict['num_event_types_pad'] = 11
    model_config_dict['event_pad_index'] = 10

    model_config = ModelConfig.parse_from_yaml_config(model_config_dict)

    nhp_model = NHP(model_config)

    print(nhp_model.__dict__)

    # config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
    #
    # model_runner = Runner.build_from_config(config)
    #
    # model_runner.run()


if __name__ == '__main__':
    main()
