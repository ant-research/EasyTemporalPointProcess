from easy_tpp.config_factory.base_config import Config
from easy_tpp.config_factory.runner_config import RunnerConfig
from easy_tpp.utils import parse_uri_to_protocol_and_path, py_assert


class HPOConfig(Config):
    def __init__(self, framework_id, storage_uri, is_continuous, num_trials, num_jobs):
        """Initialize the HPO Config

        Args:
            framework_id (str): hpo framework id.
            storage_uri (str): result storage dir.
            is_continuous (bool): whether to continuously do the optimization.
            num_trials (int): num of trails used in optimization.
            num_jobs (int): num of the jobs.
        """
        self.framework_id = framework_id or 'optuna'
        self.is_continuous = is_continuous if is_continuous is not None else True
        self.num_trials = num_trials or 50
        self.storage_uri = storage_uri
        self.num_jobs = num_jobs if num_jobs is not None else 1

    @property
    def storage_protocol(self):
        """Get the storage protocol

        Returns:
            str: the dir of the storage protocol.
        """
        storage_protocol, _ = parse_uri_to_protocol_and_path(self.storage_uri)
        return storage_protocol

    @property
    def storage_path(self):
        """Get the storage protocol

        Returns:
            str: the dir of the hpo data storage.
        """
        _, storage_path = parse_uri_to_protocol_and_path(self.storage_uri)
        return storage_path

    def get_yaml_config(self):
        """Return the config in dict (yaml compatible) format.

        Returns:
            dict: config of the HPO specs in dict format.
        """
        return {
            'framework_id': self.framework_id,
            'storage_uri': self.storage_uri,
            'is_continuous': self.is_continuous,
            'num_trials': self.num_trials,
            'num_jobs': self.num_jobs
        }

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.HPOConfig: Config class for HPO specs.
        """
        if yaml_config is None:
            return None
        else:
            return HPOConfig(
                framework_id=yaml_config.get('framework_id'),
                storage_uri=yaml_config.get('storage_uri'),
                is_continuous=yaml_config.get('is_continuous'),
                num_trials=yaml_config.get('num_trials'),
                num_jobs=yaml_config.get('num_jobs'),
            )

    def copy(self):
        """Copy the config.

        Returns:
            EasyTPP.HPOConfig: a copy of current config.
        """
        return HPOConfig(
            framework_id=self.framework_id,
            storage_uri=self.storage_uri,
            is_continuous=self.is_continuous,
            num_trials=self.num_trials,
            num_jobs=self.num_jobs
        )


@Config.register('hpo_runner_config')
class HPORunnerConfig(Config):
    def __init__(self, hpo_config, runner_config):
        """Initialize the config class

        Args:
            hpo_config (EasyTPP.HPOConfig): hpo config class.
            runner_config (EasyTPP.RunnerConfig): runner config class.
        """
        self.hpo_config = hpo_config
        self.runner_config = runner_config

    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        """Parse from the yaml to generate the config object.

        Args:
            yaml_config (dict): configs from yaml file.

        Returns:
            EasyTPP.HPORunnerConfig: Config class for HPO specs.
        """
        runner_config = RunnerConfig.parse_from_yaml_config(yaml_config, **kwargs)
        hpo_config = HPOConfig.parse_from_yaml_config(yaml_config.get('hpo'), **kwargs)
        py_assert(hpo_config is not None, ValueError, 'No hpo configs is provided for HyperTuner')
        return HPORunnerConfig(
            hpo_config=hpo_config,
            runner_config=runner_config
        )

    def copy(self):
        """Copy the config.

        Returns:
            EasyTPP.HPORunnerConfig: a copy of current config.
        """
        return HPORunnerConfig(
            hpo_config=self.hpo_config,
            runner_config=self.runner_config
        )
