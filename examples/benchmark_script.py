from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    """ Run the models on one dataset - take taxi dataset for example """

    # Run RMTPP
    config = Config.build_from_yaml_file('configs/experiment_config.yaml', experiment_id='RMTPP_train')

    model_runner = Runner.build_from_config(config)

    model_runner.run()

    # Run NHP
    config = Config.build_from_yaml_file('configs/experiment_config.yaml', experiment_id='NHP_train')

    model_runner = Runner.build_from_config(config)

    model_runner.run()

    # Run SAHP
    config = Config.build_from_yaml_file('configs/experiment_config.yaml', experiment_id='SAHP_train')

    model_runner = Runner.build_from_config(config)

    model_runner.run()

    # Run THP
    config = Config.build_from_yaml_file('configs/experiment_config.yaml', experiment_id='THP_train')

    model_runner = Runner.build_from_config(config)

    model_runner.run()

    #Run AttNHP
    #converge slow
    config = Config.build_from_yaml_file('configs/experiment_config.yaml', experiment_id='AttNHP_train')

    model_runner = Runner.build_from_config(config)

    model_runner.run()

    # Run ODETPP
    config = Config.build_from_yaml_file('configs/experiment_config.yaml', experiment_id='ODETPP_train')

    model_runner = Runner.build_from_config(config)

    model_runner.run()

    return


if __name__ == '__main__':
    main()
