from abc import abstractmethod
from typing import Any

from easy_tpp.utils import save_yaml_config, load_yaml_config, Registrable, logger


class Config(Registrable):

    def save_to_yaml_file(self, fn):
        """Save the config into the yaml file 'fn'.

        Args:
            fn (str): Target filename.

        Returns:
        """
        yaml_config = self.get_yaml_config()
        save_yaml_config(fn, yaml_config)

    @staticmethod
    def build_from_yaml_file(yaml_fn, **kwargs):
        """Load yaml config file from disk.

        Args:
            yaml_fn (str): Path of the yaml config file.

        Returns:
            EasyTPP.Config: Config object corresponding to cls.
        """
        config = load_yaml_config(yaml_fn)
        pipeline_config = config.get('pipeline_config_id')
        config_cls = Config.by_name(pipeline_config.lower())
        logger.critical(f'Load pipeline config class {config_cls.__name__}')
        return config_cls.parse_from_yaml_config(config, **kwargs)

    @abstractmethod
    def get_yaml_config(self):
        """Get the yaml format config from self.

        Returns:
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_from_yaml_config(yaml_config):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.Config: Config class for data.
        """
        pass

    @abstractmethod
    def copy(self):
        """Get a same and freely modifiable copy of self.

        Returns:
        """
        pass

    def __str__(self):
        """Str representation of the config.

        Returns:
            str: str representation of the dict format of the config.
        """
        return str(self.get_yaml_config())

    def update(self, config):
        """Update the config.

        Args:
            config (dict): config dict.

        Returns:
            EasyTPP.Config: Config class for data.
        """
        logger.critical(f'Update config class {self.__class__.__name__}')
        return self.parse_from_yaml_config(config)

    def pop(self, key: str, default_var: Any):
        """pop out the key-value item from the config.

        Args:
            key (str): key name.
            default_var (Any): default value to pop.

        Returns:
            Any: value to pop.
        """
        return vars(self).pop(key) or default_var

    def get(self, key: str, default_var: Any):
        """Retrieve the key-value item from the config.

        Args:
            key (str): key name.
            default_var (Any): default value to pop.

        Returns:
            Any: value to get.
        """
        return vars(self)[key] or default_var

    def set(self, key: str, var_to_set: Any):
        """Set the key-value item from the config.

        Args:
            key (str): key name.
            var_to_set (Any): default value to pop.

        Returns:
            Any: value to get.
        """
        vars(self)[key] = var_to_set
