import logging
import os
import sys

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.hpo.base_hpo import HyperTuner
from easy_tpp.preprocess import TPPDataLoader
from easy_tpp.runner import Runner
from easy_tpp.utils import Timer, dict_deep_update
from easy_tpp.utils.log_utils import get_logger

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
logger = get_logger('optuna_hpo')


@HyperTuner.register(name='optuna')
class OptunaTuner(HyperTuner):
    def __init__(self, config, trial_end_callbacks):
        """Initialize the Optuna Tuner class.

        Args:
            config (EasyTPP.Config): config class.
            trial_end_callbacks (list): list of trial callbacks.
        """
        super(OptunaTuner, self).__init__(config, trial_end_callbacks)

        # fetch db storage from the given storage_uri
        self.storage_fn = self._fetch_storage()

        # optuna db storage uri
        self.storage = 'sqlite:///{}'.format(self.storage_fn) if self.storage_fn else None

        self.runner_config = self.config.runner_config
        self.hpo_config = self.config.hpo_config

        # build data reader
        data_config = self.runner_config.data_config
        backend = self.runner_config.base_config.backend
        kwargs = self.runner_config.trainer_config.get_yaml_config()
        self._data_loader = TPPDataLoader(
            data_config=data_config,
            backend=backend,
            **kwargs
        )

    def get_all_best_runner_configs(self):
        """Get all best runner configs. Obtain from storage.

        Returns:
            Dict[str, EasyTPP.RunnerConfig]: Dict of all best runner configs.
        """
        runner_configs = {}
        for study_summary in optuna.get_all_study_summaries(self.storage):
            runner_configs[study_summary.study_name] = self._build_runner_config_from_storage(
                study=study_summary,
                trial=study_summary.best_trial
            )
        return runner_configs

    def get_best_runner_config_by_name(self, exp_id):
        """Get the best runner config by runner_id. Obtain it from storage.

        Args:
            exp_id (str): experiment id.

        Returns:
            EasyTPP.RunnerConfig: best runner config.
        """
        for study_summary in optuna.get_all_study_summaries(self.storage):
            if exp_id == study_summary.study_name:
                return self._build_runner_config_from_storage(study_summary, study_summary.best_trial)
        return None

    def get_num_remain_trials_by_name(self, exp_id):
        """Get the num of remaining trails by experiment id.

        Args:
            exp_id (str): experiment id.

        Returns:
            int: num of remaining trails.
        """
        for study_summary in optuna.get_all_study_summaries(self.storage):
            if exp_id == study_summary.study_name:
                study = optuna.load_study(study_name=exp_id, storage=self.storage)
                num_completed_trials = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
                num_remain_trials = self.hpo_config.num_trials - num_completed_trials
                return num_remain_trials
        return self.hpo_config.num_trials

    def optimize(
            self,
            base_runner_config,
            train_loader,
            valid_loader,
            test_loader=None,
            exp_id=None,
            **kwargs
    ):
        """Run the optimization process.

        Args:
            base_runner_config (EasyTPP.RunnerConfig): runner config.
            train_loader (EasyTPP.DataLoader): train data loader.
            valid_loader (EasyTPP.DataLoader): valid data loader
            test_loader (EasyTPP.DataLoader, optional): test data loader. Defaults to None.
            exp_id (str, optional): experiment id. Defaults to None.

        Raises:
            RuntimeError: best trial is not found.

        Returns:
            tuple: best_metric and best_runner_config
        """
        # obtain parameters
        storage = self.storage
        load_if_exists = self.hpo_config.is_continuous
        metric_direction = base_runner_config.get_metric_direction()

        # delete the study if it already existed when 'is_continue' is false
        if not load_if_exists and exp_id is not None:
            if exp_id in [std_summary.study_name for std_summary in optuna.get_all_study_summaries(storage)]:
                optuna.delete_study(
                    study_name=exp_id,
                    storage=storage
                )
        # create hpo study
        study = optuna.create_study(
            storage=storage,
            direction=metric_direction,
            load_if_exists=load_if_exists,
            study_name=exp_id,
            sampler=TPESampler(seed=9899),
        )

        # set user_attr to study
        study.set_user_attr('data_config', base_runner_config.data_config.get_yaml_config())

        # calculate the number of remaining trials
        num_completed_trials = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        num_remain_trials = self.hpo_config.num_trials - num_completed_trials
        if num_remain_trials > 0:
            logger.info(f'Number of hpo trials completed for runner {exp_id}: '
                        f'{num_completed_trials}/{self.hpo_config.num_trials}')
            objective_func = self._get_objective_func(
                base_runner_config=base_runner_config,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                **kwargs
            )
            # hpo optimize
            study.optimize(
                objective_func,
                n_trials=num_remain_trials,
                callbacks=[self._optimize_trial_end],
                gc_after_trial=True,
                n_jobs=self.hpo_config.num_jobs,
            )

        # statistics of this hpo
        pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
        complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        logger.info("HPO - Optuna statistics:")
        logger.info(f"\tNumber of finished trials: {len(study.trials)}")
        logger.info(f"\tNumber of pruned trials: {len(pruned_trials)}")
        logger.info(f"\tNumber of complete trials: {len(complete_trials)}")

        if len(study.best_trials) == 0:
            raise RuntimeError('Best trial is not found, please check the model or metric.')
        trial = study.best_trial
        logger.info(f"HPO - Best metric value ({metric_direction}): {trial.value}")

        logger.info(f"Best Parameters: ")
        for key, value in trial.params.items():
            logger.info(f"\t{key}: {value}")

        best_metric = trial.value
        best_runner_config = RunnerConfig.parse_from_yaml_config(
            trial.user_attrs['runner_config'],
            data_config=base_runner_config.data_config,
        )
        return best_metric, best_runner_config

    def _get_objective_func(
            self,
            base_runner_config,
            **kwargs,
    ):
        """Get the optimization objective function.

        Args:
            base_runner_config (EasyTPP.Config): runner config.

        Raises:
            e: RuntimeError
            optuna.TrialPruned: message in trials.

        Returns:
            _type_: _description_
        """
        trial_func = self.retrieve_trial_func_by_model_name(base_runner_config.base_config.exp_id)

        def objective(trial):
            timer = Timer()
            timer.start()
            logger.info(f'Start the trial {trial.number} ...')
            # get a copy of base runner config for isolation
            # generate new runner  runners
            runner_config = base_runner_config.copy()

            trial_model_info = trial_func(
                trial,
                trainder_config=runner_config.trainer_config,
                model_config=runner_config.model_config
            )

            # use predefined trial to update model_info
            runner_config_dict = dict_deep_update(
                target=runner_config.get_yaml_config(),
                source=trial_model_info,
                is_add_new_key=False)

            # eval the "suggest" in runner_config (actually run trial suggestion)
            runner_config_dict = self._eval_str_trial_to_dict(trial,
                                                              runner_config_dict)

            runner_config = RunnerConfig.parse_from_yaml_config(runner_config_dict, direct_parse=True)

            runner = Runner.build_from_config(
                runner_config=runner_config,
                unique_model_dir=True,
                skip_data_loader=True,
            )
            try:
                # train model
                runner.train(**kwargs)
                # evaluate model
                metric = runner.evaluate(trial=trial, **kwargs)
                # save the final model
                runner.save()
            except RuntimeError as e:
                # add the error message into trial
                err_msg = str(e)
                trial.set_user_attr("error", err_msg)

                logger.error(f'Error in the trial {trial.number}: {err_msg}')

                # just prune the errors like 'out of memory'
                if 'out of memory' not in err_msg:
                    raise e
                raise optuna.TrialPruned()
            finally:
                # add model path into trial
                trial.set_user_attr("model_dir", runner.runner_config.model_dir)
                # trial.set_user_attr("model_config", runner.runner_config.model_config)
                trial.set_user_attr("runner_config", runner.runner_config.get_yaml_config())

                logger.info(f'End trial {trial.number} ! Cost time: {timer.end()}')

            return metric

        return objective

    def _optimize_trial_end(self, study, trial):
        """End the process of trials.

        Args:
            study (optuna.study.Study): an object of optuna :class:`~optuna.study` to be studied during optimization.
            trial (optuna.trial.FrozenTrial): an object of optuna :class:`~optuna.trial` that stores trial information.
        """
        # push storage to the specified uri
        self._push_storage(trial)

        is_best_yet = (trial in study.best_trials)

        runner_id = study.study_name
        runner_config = self._build_runner_config_from_storage(study, trial)

        # invoke callbacks
        for callback in self.trial_end_callbacks:
            callback(runner_id, runner_config, is_best_yet)

        # clean disk
        if not is_best_yet and os.path.exists(trial.user_attrs['model_dir']):
            os.system(f"rm -fr {trial.user_attrs['model_dir']}")

    def _eval_str_trial_to_dict(self, trial, a_dict):
        for key, val in a_dict.items():
            if type(val) == str and val.startswith('suggest_'):
                idx = val.find('(')
                prefix = val[:idx + 1]
                suffix = val[idx + 1:]

                # get trial variable name
                trial_name = [k for k, v in locals().items() if v == trial][0]

                code = '''{0}.{1}"{2}",{3}'''.format(trial_name, prefix, key, suffix)
                a_dict[key] = eval(code)
            elif type(val) == dict:
                self._eval_str_trial_to_dict(trial, a_dict[key])

        return a_dict

    def _build_runner_config_from_storage(
            self,
            study,
            trial
    ):
        """Initialize the RunnerConfig from the study and trial.

        Args:
            study (optuna.study.Study): an object of optuna :class:`~optuna.study` to be studied during optimization.
            trial (optuna.trial.FrozenTrial): an object of optuna :class:`~optuna.trial` that stores trial information.

        Returns:
            EasyTPP.Config: RunnerConfig object.
        """
        runner_config_dict = trial.user_attrs['runner_config']
        return RunnerConfig.parse_from_yaml_config(runner_config_dict, direct_parse=True)

    def run(self):
        """Run the HPO process.

        Returns:
            tuple: best_metric, best_runner_config
        """
        # to avoid to load unused data
        train_loader, valid_loader, test_loader = None, None, None
        exp_id = self.runner_config.base_config.exp_id

        if self.get_num_remain_trials_by_name(exp_id) > 0:
            train_loader = self._data_loader.get_loader(split='train')
            valid_loader = self._data_loader.get_loader(split='dev')
            if self.runner_config.data_config.test_dir is not None:
                test_loader = self._data_loader.get_loader(split='test')

        best_metric, best_runner_config = self.optimize(
            base_runner_config=self.runner_config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            exp_id=exp_id
        )

        return best_metric, best_runner_config

    def _fetch_storage(self):
        """Retrieve the stored model.

        Returns:
            str: dir of the stored model.
        """
        local_storage_fn = self.config.hpo_config.storage_path

        # return the local storage location
        return local_storage_fn

    def _push_storage(self, trial):
        """_summary_

        Args:
            trial (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        # save hpo storage to remote if it's in remote
        if self.config.hpo_config.storage_protocol == 'oss':
            raise NotImplementedError

        return
