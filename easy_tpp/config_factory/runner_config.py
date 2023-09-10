import copy
import os

from easy_tpp.config_factory.config import Config
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.config_factory.model_config import TrainerConfig, ModelConfig, BaseConfig
from easy_tpp.utils import create_folder, logger, get_unique_id, get_stage, RunnerPhase, \
    MetricsHelper, DefaultRunnerConfig, py_assert, is_torch_available, is_tf_available, is_tf_gpu_available, \
    is_torch_gpu_available
from easy_tpp.utils.const import Backend


@Config.register('runner_config')
class RunnerConfig(Config):
    def __init__(self, base_config, model_config, data_config, trainer_config):
        """Initialize the Config class.

        Args:
            base_config (EasyTPP.BaseConfig): BaseConfig object.
            model_config (EasyTPP.ModelConfig): ModelConfig object.
            data_config (EasyTPP.DataConfig): DataConfig object.
            trainer_config (EasyTPP.TrainerConfig): TrainerConfig object
        """
        self.data_config = data_config
        self.model_config = model_config
        self.base_config = base_config
        self.trainer_config = trainer_config

        self.ensure_valid_config()
        self.update_config()

        # save the complete config
        save_dir = self.base_config.specs['output_config_dir']
        self.save_to_yaml_file(save_dir)

        logger.info(f'Save the config to {save_dir}')

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the runner config in dict format.
        """
        return {'data_config': self.data_config.get_yaml_config(),
                'base_config': self.base_config.get_yaml_config(),
                'model_config': self.model_config.get_yaml_config(),
                'trainer_config': self.trainer_config.get_yaml_config()}

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            RunnerConfig: Config class for trainer specs.
        """
        direct_parse = kwargs.get('direct_parse', False)
        if not direct_parse:
            exp_id = kwargs.get('experiment_id')
            yaml_exp_config = yaml_config[exp_id]
            dataset_id = yaml_exp_config.get('base_config').get('dataset_id')
            if dataset_id is None:
                dataset_id = DefaultRunnerConfig.DEFAULT_DATASET_ID
            try:
                yaml_data_config = yaml_config['data'][dataset_id]
            except KeyError:
                raise RuntimeError('dataset_id={} is not found in config.'.format(dataset_id))

            data_config = DataConfig.parse_from_yaml_config(yaml_data_config)
            # add exp id to base config
            yaml_exp_config.get('base_config').update(exp_id=exp_id)

        else:
            yaml_exp_config = yaml_config
            data_config = DataConfig.parse_from_yaml_config(yaml_exp_config.get('data_config'))

        base_config = BaseConfig.parse_from_yaml_config(yaml_exp_config.get('base_config'))
        model_config = ModelConfig.parse_from_yaml_config(yaml_exp_config.get('model_config'))
        trainer_config = TrainerConfig.parse_from_yaml_config(yaml_exp_config.get('trainer_config'))

        return RunnerConfig(
            data_config=data_config,
            base_config=base_config,
            model_config=model_config,
            trainer_config=trainer_config
        )

    def ensure_valid_config(self):
        """Do some sanity check about the config, to avoid conflicts in settings.
        """

        # during testing we dont do shuffle by default
        self.trainer_config.shuffle = False

        # during testing we dont apply tfb by default
        self.trainer_config.use_tfb = False

        return

    def update_config(self):
        """Updated config dict.
        """
        model_folder_name = get_unique_id()

        log_folder = create_folder(self.base_config.base_dir, model_folder_name)
        model_folder = create_folder(log_folder, 'models')

        self.base_config.specs['log_folder'] = log_folder
        self.base_config.specs['saved_model_dir'] = os.path.join(model_folder, 'saved_model')
        self.base_config.specs['saved_log_dir'] = os.path.join(log_folder, 'log')
        self.base_config.specs['output_config_dir'] = os.path.join(log_folder,
                                                                   f'{self.base_config.exp_id}_output.yaml')

        if self.trainer_config.use_tfb:
            self.base_config.specs['tfb_train_dir'] = create_folder(log_folder, 'tfb_train')
            self.base_config.specs['tfb_valid_dir'] = create_folder(log_folder, 'tfb_valid')

        current_stage = get_stage(self.base_config.stage)
        is_training = current_stage == RunnerPhase.TRAIN
        self.model_config.is_training = is_training
        self.model_config.gpu = self.trainer_config.gpu

        # update the dataset config => model config
        self.model_config.num_event_types_pad = self.data_config.data_specs.num_event_types_pad
        self.model_config.num_event_types = self.data_config.data_specs.num_event_types
        self.model_config.pad_token_id = self.data_config.data_specs.pad_token_id
        self.model_config.max_len = self.data_config.data_specs.max_len

        # update base config => model config
        model_id = self.base_config.model_id
        self.model_config.model_id = model_id

        if self.base_config.model_id == 'ODETPP' and self.base_config.backend == Backend.TF:
            py_assert(self.data_config.data_specs.padding_strategy == 'max_length',
                      ValueError,
                      'For ODETPP in TensorFlow, we must pad all sequence to '
                      'the same length (max len of the sequences)!')

        run = current_stage
        use_torch = self.base_config.backend == Backend.Torch
        device = 'GPU' if self.trainer_config.gpu >= 0 else 'CPU'

        py_assert(is_torch_available() if use_torch else is_tf_available(), ValueError,
                  f'Backend {self.base_config.backend} is not supported in the current environment yet !')

        if use_torch and device == 'GPU':
            py_assert(is_torch_gpu_available(),
                      ValueError,
                      f'Torch cuda is not supported in the current environment yet!')

        if not use_torch and device == 'GPU':
            py_assert(is_tf_gpu_available(),
                      ValueError,
                      f'Tensorflow GPU is not supported in the current environment yet!')

        critical_msg = '{run} model {model_name} using {device} ' \
                       'with {tf_torch} backend'.format(run=run,
                                                        model_name=model_id,
                                                        device=device,
                                                        tf_torch=self.base_config.backend)

        logger.critical(critical_msg)

        return

    def get_metric_functions(self):
        return MetricsHelper.get_metrics_callback_from_names(self.trainer_config.metrics)

    def get_metric_direction(self, metric_name='rmse'):
        return MetricsHelper.get_metric_direction(metric_name)

    def copy(self):
        """Copy the config.

        Returns:
            RunnerConfig: a copy of current config.
        """
        return RunnerConfig(
            base_config=copy.deepcopy(self.base_config),
            model_config=copy.deepcopy(self.model_config),
            data_config=copy.deepcopy(self.data_config),
            trainer_config=copy.deepcopy(self.trainer_config)
        )
