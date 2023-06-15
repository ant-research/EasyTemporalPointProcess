from abc import abstractmethod
from collections import defaultdict
from typing import List

from easy_tpp.utils import logger, Registrable


class HyperTuner(Registrable):
    _trial_register_center = defaultdict(dict)

    def __init__(self, config, trial_end_callbacks: List[callable] = None):
        """Initialize the tuner

        Args:
            config (EasyTPP.Config): config class
            trial_end_callbacks (List[callable]): List of callback functions to be executed after each trial.
        """
        self.config = config
        self.trial_end_callbacks = trial_end_callbacks or []
        logger.info(f'Storage of hpo framework: {self.config.hpo_config.storage_uri}')

    @abstractmethod
    def get_all_best_runner_configs(self):
        pass

    @abstractmethod
    def get_best_runner_config_by_name(self, runner_id):
        """

        Args:
            runner_id (str):

        Returns:

        """
        pass

    @abstractmethod
    def get_num_remain_trials_by_name(self, runner_id):
        pass

    @staticmethod
    def build_from_config(config, trial_end_callbacks: List[callable] = None):
        """Load yaml config file from disk.

        Args:
            config (EasyTPP.Config): config class
            trial_end_callbacks (List[callable]): List of callback functions to be executed after each trial.

        Returns:
            EasyTPP.Config: Config object corresponding to cls.
        """
        runner_cls = HyperTuner.by_name(config.hpo_config.framework_id)
        return runner_cls(config, trial_end_callbacks)

    # ---------------------- Trail Register and Get Functions ---------------------

    @classmethod
    def register_trial_func(cls, model_id, overwrite=True):
        """Register the trial functions in HPO

        Args:
            model_id (str): id of the models.
            overwrite (bool, optional): whether to overwrite the trial function. Defaults to True.

        Returns:
            dict: the registered trial function
        """
        register_center = HyperTuner._trial_register_center

        def _register_trial(func):
            if model_id in register_center[cls]:
                if overwrite:
                    register_center[cls][model_id] = func
                    logger.info(f'The trial for {model_id} is already registered, but overwrite it.')
                else:
                    logger.warn(f'The trial for {model_id} is already registered, and cannot be overwritten!')
            else:
                register_center[cls][model_id] = func
                logger.info(f'Trial register: {cls.get_registered_name()} - {model_id}')
            return func

        return _register_trial

    @classmethod
    def retrieve_trial_func_by_model_name(cls, name):
        """Retrieve the trail function by the model id

        Args:
            name (str): model id.

        Raises:
            RuntimeError: non registered error for the hpo framework.

        Returns:
            dict: registered trial center
        """
        cls_trial_rc = HyperTuner._trial_register_center[cls]
        if name not in cls_trial_rc:
            if 'default' in cls_trial_rc:
                logger.warn(
                    f'Trial for {name} in {cls.get_registered_name()} is not existed, and use default trial!'
                )
                name = 'default'
            else:
                raise RuntimeError(f'This HPO Framework is not registered!')
        return cls_trial_rc[name]

    @classmethod
    def get_registered_name(cls):
        """Get the name of the registered hpo class.

        Returns:
            str: the name of the registered hpo class.
        """
        hpo_rc = HyperTuner.registry_dict()
        for registered_name, hpo_cls in hpo_rc.items():
            if cls in hpo_cls:
                return registered_name

        logger.warn(f'The hpo framework is not registered: {cls}')
        return None

    @abstractmethod
    def run(self):
        """Run the process.

        Raises:
            NotImplementedError: error raised in base class.
        """
        raise NotImplementedError
