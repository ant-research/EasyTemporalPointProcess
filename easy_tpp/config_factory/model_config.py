from easy_tpp.config_factory.config import Config

from easy_tpp.utils.const import Backend


class TrainerConfig(Config):

    def __init__(self, **kwargs):
        """Initialize the Config class.
        """
        self.seed = kwargs.get('seed', 9899)
        self.gpu = kwargs.get('gpu', -1)
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
        return {'seed': self.seed,
                'gpu': self.gpu,
                'batch_size': self.batch_size,
                'max_epoch': self.max_epoch,
                'shuffle': self.shuffle,
                'optimizer': self.optimizer,
                'learning_rate': self.learning_rate,
                'valid_freq': self.valid_freq,
                'use_tfb': self.use_tfb,
                'metrics': self.metrics
                }

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.TrainerConfig: Config class for trainer specs.
        """
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
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.ThinningConfig: Config class for thinning algorithms.
        """
        return ThinningConfig(**yaml_config) if yaml_config is not None else None

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
        else:
            raise ValueError(
                f"Backend should be 'torch' or 'pytorch', current value: {backend}"
            )

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the base config specs in dict format.
        """
        return {'stage': self.stage,
                'backend': str(self.backend),
                'dataset_id': self.dataset_id,
                'runner_id': self.runner_id,
                'model_id': self.model_id,
                'base_dir': self.base_dir,
                'specs': self.specs}

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            BaseConfig: Config class for trainer specs.
        """
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
        self.sharing_param_layer = kwargs.get('sharing_param_layer', False)
        self.use_mc_samples = kwargs.get('use_mc_samples', True)  # if using MC samples in computing log-likelihood
        self.loss_integral_num_sample_per_step = kwargs.get('loss_integral_num_sample_per_step', 20)  # mc_num_sample_per_step
        self.dropout_rate = kwargs.get('dropout_rate', 0.0)
        self.use_ln = kwargs.get('use_ln', False)
        self.thinning = ThinningConfig.parse_from_yaml_config(kwargs.get('thinning'))
        self.is_training = kwargs.get('training', False)
        self.num_event_types_pad = kwargs.get('num_event_types_pad', None)
        self.num_event_types = kwargs.get('num_event_types', None)
        self.pad_token_id = kwargs.get('event_pad_index', None)
        self.model_id = kwargs.get('model_id', None)
        self.pretrained_model_dir = kwargs.get('pretrained_model_dir', None)
        self.gpu = kwargs.get('gpu', -1)
        self.model_specs = kwargs.get('model_specs', {})

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the model config specs in dict format.
        """
        return {'rnn_type': self.rnn_type,
                'hidden_size': self.hidden_size,
                'time_emb_size': self.time_emb_size,
                'num_layers': self.num_layers,
                'sharing_param_layer': self.sharing_param_layer,
                'loss_integral_num_sample_per_step': self.loss_integral_num_sample_per_step,
                'dropout_rate': self.dropout_rate,
                'use_ln': self.use_ln,
                # for some models / cases we may not need to pass thinning config
                # e.g., for intensity-free model
                'thinning': None if self.thinning is None else self.thinning.get_yaml_config(),
                'num_event_types_pad': self.num_event_types_pad,
                'num_event_types': self.num_event_types,
                'event_pad_index': self.pad_token_id,
                'model_id': self.model_id,
                'pretrained_model_dir': self.pretrained_model_dir,
                'gpu': self.gpu,
                'model_specs': self.model_specs}

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            ModelConfig: Config class for trainer specs.
        """
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
                           sharing_param_layer=self.sharing_param_layer,
                           loss_integral_num_sample_per_step=self.loss_integral_num_sample_per_step,
                           dropout_rate=self.dropout_rate,
                           use_ln=self.use_ln,
                           thinning=self.thinning,
                           num_event_types_pad=self.num_event_types_pad,
                           num_event_types=self.num_event_types,
                           event_pad_index=self.pad_token_id,
                           pretrained_model_dir=self.pretrained_model_dir,
                           gpu=self.gpu,
                           model_specs=self.model_specs)
