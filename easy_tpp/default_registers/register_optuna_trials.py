from easy_tpp.hpo.optuna_hpo import OptunaTuner


@OptunaTuner.register_trial_func(model_id='default', overwrite=False)
def default_trial(trial, **kwargs):
    setting = {
        "trainer_config": {"max_epoch": "suggest_int(40, 100, log=True)",
                           "batch_size": 256,
                           "optimizer": "adam",
                           "learning_rate": "suggest_float(5e-4, 1e-2, log=True)"},
        "model_config": {"hidden_size": "suggest_int(16, 32)"}
    }
    return setting
