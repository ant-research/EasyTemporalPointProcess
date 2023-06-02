import copy
import os

from easy_tpp.config_factory.base_config import Config
from easy_tpp.config_factory.data_config import DataConfig
from easy_tpp.utils import create_folder, logger, get_unique_id, get_stage, RunnerPhase, \
    MetricsHelper, DefaultRunnerConfig, py_assert, is_torch_available, is_tf_available
from easy_tpp.utils.const import Backend


class TrainerConfig(Config):

    def __init__(self, **kwargs):
        """Initialize the Config class.
        """
        self.batch_size = kwargs.get('batch_size', 256)
        self.max_epoch = kwargs.get('max_epoch', 10)
        self.shuffle = kwargs.get('shuffle', False)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.learning_rate = kwargs.get('learning_rate', 1.e-3)
        self.valid_freq = kwargs.get('valid_freq', 1)
        self.use_tfb = kwargs.get('use_tfb', False)
        self.metrics = kwargs.get('metrics', ['acc', 'rmse'])

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the trainer specs in dict format.
        """
        return {'batch_size': self.batch_size,
                'max_epoch': self.max_epoch,
                'shuffle': self.shuffle,
                'optimizer': self.optimizer,
                'learning_rate': self.learning_rate,
                'valid_freq': self.valid_freq,
                'use_tfb': self.use_tfb,
                'metrics': self.metrics
                }

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.TrainerConfig: Config class for trainer specs.
        """
        if yaml_config is None:
            return TrainerConfig()
        else:
            return TrainerConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            EasyTPP.TrainerConfig: a copy of current config.
        """
        return TrainerConfig(batch_size=self.batch_size,
                             max_epoch=self.max_epoch,
                             shuffle=self.shuffle,
                             optimizer=self.optimizer,
                             learning_rate=self.learning_rate,
                             valid_freq=self.valid_freq,
                             use_tfb=self.use_tfb,
                             metrics=self.metrics
                             )


class ThinningConfig(Config):
    def __init__(self, **kwargs):
        """Initialize the Config class.
        """
        self.num_seq = kwargs.get('num_seq', 10)
        self.num_sample = kwargs.get('num_sample', 1)
        self.num_exp = kwargs.get('num_exp', 500)
        self.look_ahead_time = kwargs.get('look_ahead_time', 10)
        self.patience_counter = kwargs.get('patience_counter', 5)
        self.over_sample_rate = kwargs.get('over_sample_rate', 5)
        self.num_samples_boundary = kwargs.get('num_samples_boundary', 5)
        self.dtime_max = kwargs.get('dtime_max', 5)
        # we pad the sequence at the front only in multi-step generation
        self.num_step_gen = kwargs.get('num_step_gen', 1)

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the thinning specs in dict format.
        """
        return {'num_seq': self.num_seq,
                'num_sample': self.num_sample,
                'num_exp': self.num_exp,
                'look_ahead_time': self.look_ahead_time,
                'patience_counter': self.patience_counter,
                'over_sample_rate': self.over_sample_rate,
                'num_samples_boundary': self.num_samples_boundary,
                'dtime_max': self.dtime_max,
                'num_step_gen': self.num_step_gen}

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.ThinningConfig: Config class for thinning algorithms.
        """
        if yaml_config is None:
            return None
        else:
            return ThinningConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            EasyTPP.ThinningConfig: a copy of current config.
        """
        return ThinningConfig(num_seq=self.num_seq,
                              num_sample=self.num_sample,
                              num_exp=self.num_exp,
                              look_ahead_time=self.look_ahead_time,
                              patience_counter=self.patience_counter,
                              over_sample_rate=self.over_sample_rate,
                              num_samples_boundary=self.num_samples_boundary,
                              dtime_max=self.dtime_max,
                              num_step_gen=self.num_step_gen)


class BaseConfig(Config):
    def __init__(self, **kwargs):
        """Initialize the Config class.
        """
        self.stage = kwargs.get('stage')
        self.backend = kwargs.get('backend')
        self.dataset_id = kwargs.get('dataset_id')
        self.runner_id = kwargs.get('runner_id')
        self.model_id = kwargs.get('model_id')
        self.exp_id = kwargs.get('exp_id')
        self.base_dir = kwargs.get('base_dir')
        self.specs = kwargs.get('specs', {})
        self.backend = self.set_backend(self.backend)

    @staticmethod
    def set_backend(backend):
        if backend.lower() in ['torch', 'pytorch']:
            return Backend.Torch
        elif backend.lower() in ['tf', 'tensorflow']:
            return Backend.TF
        else:
            raise ValueError(
                f"Backend  should be selected between 'torch or pytorch' and 'tf or tensorflow', "
                f"current value: {backend}"
            )

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the base config specs in dict format.
        """
        return {'stage': self.stage,
                'backend': self.backend,
                'dataset_id': self.dataset_id,
                'runner_id': self.runner_id,
                'model_id': self.model_id,
                'base_dir': self.base_dir,
                'specs': self.specs}

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            BaseConfig: Config class for trainer specs.
        """
        if yaml_config is None:
            return None
        else:
            yaml_config.update(kwargs)
            return BaseConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            BaseConfig: a copy of current config.
        """
        return BaseConfig(stage=self.stage,
                          backend=self.backend,
                          dataset_id=self.dataset_id,
                          runner_id=self.runner_id,
                          model_id=self.model_id,
                          base_dir=self.base_dir,
                          specs=self.specs)


class ModelConfig(Config):
    def __init__(self, **kwargs):
        """Initialize the Config class.
        """
        self.rnn_type = kwargs.get('rnn_type', 'LSTM')
        self.hidden_size = kwargs.get('hidden_size', 32)
        self.time_emb_size = kwargs.get('time_emb_size', 16)
        self.num_layers = kwargs.get('num_layers', 2)
        self.num_heads = kwargs.get('num_heads', 2)
        self.mc_num_sample_per_step = kwargs.get('mc_num_sample_per_step', 20)
        self.sharing_param_layer = kwargs.get('sharing_param_layer', False)
        self.loss_integral_num_sample_per_step = kwargs.get('loss_integral_num_sample_per_step', 20)
        self.dropout_rate = kwargs.get('dropout_rate', 0.0)
        self.use_ln = kwargs.get('use_ln', False)
        self.seed = kwargs.get('seed', 9899)
        self.gpu = kwargs.get('gpu', -1)
        self.trainer = TrainerConfig.parse_from_yaml_config(kwargs.get('trainer'))
        self.thinning = ThinningConfig.parse_from_yaml_config(kwargs.get('thinning'))
        self.is_training = kwargs.get('training', False)
        self.num_event_types_pad = kwargs.get('num_event_types_pad', None)
        self.num_event_types = kwargs.get('num_event_types', None)
        self.pad_token_id = kwargs.get('event_pad_index', None)
        self.model_id = kwargs.get('model_id', None)
        self.pretrained_model_dir = kwargs.get('pretrained_model_dir', None)
        self.specs = kwargs.get('model_specs', {})

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the model config specs in dict format.
        """
        return {'rnn_type': self.rnn_type,
                'hidden_size': self.hidden_size,
                'time_emb_size': self.time_emb_size,
                'num_layers': self.num_layers,
                'mc_num_sample_per_step': self.mc_num_sample_per_step,
                'sharing_param_layer': self.sharing_param_layer,
                'loss_integral_num_sample_per_step': self.loss_integral_num_sample_per_step,
                'dropout_rate': self.dropout_rate,
                'use_ln': self.use_ln,
                'seed': self.seed,
                'gpu': self.gpu,
                'trainer': self.trainer.get_yaml_config(),
                # for some models / cases we may not need to pass thinning config
                # e.g., for intensity-free model
                'thinning': None if self.thinning is None else self.thinning.get_yaml_config(),
                'num_event_types_pad': self.num_event_types_pad,
                'num_event_types': self.num_event_types,
                'event_pad_index': self.pad_token_id,
                'model_id': self.model_id,
                'pretrained_model_dir': self.pretrained_model_dir,
                'specs': self.specs}

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            ModelConfig: Config class for trainer specs.
        """
        if yaml_config is None:
            return None
        else:
            return ModelConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            ModelConfig: a copy of current config.
        """
        return ModelConfig(rnn_type=self.rnn_type,
                           hidden_size=self.hidden_size,
                           time_emb_size=self.time_emb_size,
                           num_layers=self.num_layers,
                           mc_num_sample_per_step=self.mc_num_sample_per_step,
                           sharing_param_layer=self.sharing_param_layer,
                           loss_integral_num_sample_per_step=self.loss_integral_num_sample_per_step,
                           dropout_rate=self.dropout_rate,
                           use_ln=self.use_ln,
                           seed=self.seed,
                           gpu=self.gpu,
                           trainer=self.trainer,
                           thinning=self.thinning,
                           num_event_types_pad=self.num_event_types_pad,
                           num_event_types=self.num_event_types,
                           event_pad_index=self.pad_token_id,
                           pretrained_model_dir=self.pretrained_model_dir,
                           specs=self.specs)


@Config.register('runner_config')
class RunnerConfig(Config):
    def __init__(self, base_config, model_config, data_config):
        """Initialize the Config class.

        Args:
            base_config (BaseConfig): BaseConfig object.
            model_config (ModelConfig): ModelConfig object.
            data_config (DataConfig): DataConfig object.
        """
        self.data_config = data_config
        self.model_config = model_config
        self.base_config = base_config

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
                'model_config': self.model_config.get_yaml_config()}

    @property
    def exp_id(self):
        return self.base_config.exp_id

    @property
    def model_id(self):
        return self.base_config.model_id

    @property
    def dataset_id(self):
        return self.base_config.dataset_id

    @property
    def stage(self):
        return self.base_config.stage

    @property
    def backend(self):
        return self.base_config.backend

    @property
    def runner_id(self):
        return self.base_config.runner_id

    @property
    def trainer(self):
        return self.model_config.trainer

    @property
    def model_dir(self):
        return self.base_config.specs.get('saved_model_dir')

    @model_dir.setter
    def model_dir(self, update_dir):
        self.base_config.specs['saved_model_dir'] = update_dir

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            RunnerConfig: Config class for trainer specs.
        """
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
        base_config = BaseConfig.parse_from_yaml_config(yaml_exp_config.get('base_config'), exp_id=exp_id)
        model_config = ModelConfig.parse_from_yaml_config(yaml_exp_config.get('model_config'))

        return RunnerConfig(
            data_config=data_config,
            base_config=base_config,  # add exp id to base config
            model_config=model_config,
        )

    def ensure_valid_config(self):
        """Do some sanity check about the config, to avoid conflicts in settings.
        """

        # during testing we dont do shuffle by default
        self.model_config.trainer.shuffle = False

        # during testing we dont apply tfb by default
        self.model_config.trainer.use_tfb = False

        return

    def update_config(self):
        """Updated config dict.
        """
        model_folder_name = get_unique_id()

        self.log_folder = create_folder(self.base_config.base_dir, model_folder_name)
        self.model_folder = create_folder(self.log_folder, 'models')

        self.base_config.specs['log_folder'] = self.log_folder
        self.base_config.specs['saved_model_dir'] = os.path.join(self.model_folder, 'saved_model')
        self.base_config.specs['saved_log_dir'] = os.path.join(self.log_folder, 'log')
        self.base_config.specs['output_config_dir'] = os.path.join(self.log_folder, f'{self.exp_id}_output.yaml')

        if self.model_config.trainer.use_tfb:
            self.base_config.specs['tfb_train_dir'] = create_folder(self.log_folder, 'tfb_train')
            self.base_config.specs['tfb_valid_dir'] = create_folder(self.log_folder, 'tfb_valid')

        current_stage = get_stage(self.stage)
        is_training = current_stage == RunnerPhase.TRAIN
        self.model_config.is_training = is_training

        # update the dataset config => model config
        self.model_config.num_event_types_pad = self.data_config.specs.num_event_types_pad
        self.model_config.num_event_types = self.data_config.specs.num_event_types
        self.model_config.pad_token_id = self.data_config.specs.pad_token_id
        self.model_config.max_len = self.data_config.specs.max_len

        # update base config => model config
        model_id = self.base_config.model_id
        self.model_config.model_id = model_id

        if self.model_id == 'ODETPP' and self.backend == Backend.TF:
            py_assert(self.data_config.specs.padding_strategy == 'max_length',
                      ValueError,
                      'For ODETPP in TensorFlow, we must pad all sequence to '
                      'the same length (max len of the sequences)!')

        run = current_stage
        tf_torch = self.backend == Backend.Torch
        device = 'GPU' if self.model_config.gpu >= 0 else 'CPU'

        py_assert(is_torch_available() if tf_torch else is_tf_available(), EnvironmentError,
                  f'Backend {tf_torch} not supported yet!')

        critical_msg = '{run} model {model_name} using {device} with {tf_torch} backend'.format(run=run,
                                                                                                model_name=model_id,
                                                                                                device=device,
                                                                                                tf_torch=self.backend)

        logger.critical(critical_msg)

        return

    def get_metric_functions(self):
        return MetricsHelper.get_metrics_callback_from_names(self.trainer.metrics)

    def get_metric_direction(self):
        return MetricsHelper.get_metric_direction('loglike')

    def copy(self):
        """Copy the config.

        Returns:
            RunnerConfig: a copy of current config.
        """
        return RunnerConfig(
            base_config=copy.deepcopy(self.base_config),
            model_config=copy.deepcopy(self.model_config),
            data_config=copy.deepcopy(self.data_config),
        )
