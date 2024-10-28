from easy_tpp.config_factory.config import Config


class DataSpecConfig(Config):
    def __init__(self, **kwargs):
        """Initialize the Config class.
        """
        self.num_event_types = kwargs.get('num_event_types')
        self.pad_token_id = kwargs.get('pad_token_id')
        self.padding_side = kwargs.get('padding_side')
        self.truncation_side = kwargs.get('truncation_side')
        self.padding_strategy = kwargs.get('padding_strategy')
        self.max_len = kwargs.get('max_len')
        self.truncation_strategy = kwargs.get('truncation_strategy')
        self.num_event_types_pad = self.num_event_types + 1
        self.model_input_names = kwargs.get('model_input_names')

        if self.padding_side is not None and self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        if self.truncation_side is not None and self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the data specs in dict format.
        """
        return {
            'num_event_types': self.num_event_types,
            'pad_token_id': self.pad_token_id,
            'padding_side': self.padding_side,
            'truncation_side': self.truncation_side,
            'padding_strategy': self.padding_strategy,
            'truncation_strategy': self.truncation_strategy,
            'max_len': self.max_len
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            DataSpecConfig: Config class for data specs.
        """
        return DataSpecConfig(**yaml_config)

    def copy(self):
        """Copy the config.

        Returns:
            DataSpecConfig: a copy of current config.
        """
        return DataSpecConfig(num_event_types_pad=self.num_event_types_pad,
                              num_event_types=self.num_event_types,
                              event_pad_index=self.pad_token_id,
                              padding_side=self.padding_side,
                              truncation_side=self.truncation_side,
                              padding_strategy=self.padding_strategy,
                              truncation_strategy=self.truncation_strategy,
                              max_len=self.max_len)


@Config.register('data_config')
class DataConfig(Config):
    def __init__(self, train_dir, valid_dir, test_dir, data_format, specs=None):
        """Initialize the DataConfig object.

        Args:
            train_dir (str): dir of tran set.
            valid_dir (str): dir of valid set.
            test_dir (str): dir of test set.
            specs (dict, optional): specs of dataset. Defaults to None.
        """
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.data_specs = specs or DataSpecConfig()
        self.data_format = train_dir.split('.')[-1] if data_format is None else data_format

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the data in dict format.
        """
        return {
            'train_dir': self.train_dir,
            'valid_dir': self.valid_dir,
            'test_dir': self.test_dir,
            'data_format': self.data_format,
            'data_specs': self.data_specs.get_yaml_config(),
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.DataConfig: Config class for data.
        """
        return DataConfig(
            train_dir=yaml_config.get('train_dir'),
            valid_dir=yaml_config.get('valid_dir'),
            test_dir=yaml_config.get('test_dir'),
            data_format=yaml_config.get('data_format'),
            specs=DataSpecConfig.parse_from_yaml_config(yaml_config.get('data_specs'))
        )

    def copy(self):
        """Copy the config.

        Returns:
            EasyTPP.DataConfig: a copy of current config.
        """
        return DataConfig(train_dir=self.train_dir,
                          valid_dir=self.valid_dir,
                          test_dir=self.test_dir,
                          specs=self.data_specs)

    def get_data_dir(self, split):
        """Get the dir of the source raw data.

        Args:
            split (str): dataset split notation, 'train', 'dev' or 'valid', 'test'.

        Returns:
            str: dir of the source raw data file.
        """
        split = split.lower()
        if split == 'train':
            return self.train_dir
        elif split in ['dev', 'valid']:
            return self.valid_dir
        else:
            return self.test_dir
