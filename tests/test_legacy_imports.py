def test_flat_model_imports_and_aliases():
    from easy_tpp.model import (
        ANHN,
        AttNHP,
        BaseModel,
        EventSampler,
        FullyNN,
        IntensityFree,
        NHP,
        ODETPP,
        RMTPP,
        S2P2,
        SAHP,
        THP,
        TorchANHN,
        TorchAttNHP,
        TorchBaseModel,
        TorchFullyNN,
        TorchIntensityFree,
        TorchNHP,
        TorchODETPP,
        TorchRMTPP,
        TorchS2P2,
        TorchSAHP,
        TorchTHP,
        TorchWSMTHP,
        WSMTHP,
    )
    from easy_tpp.model.intensity_free import LogNormalMixtureDistribution

    alias_pairs = (
        (ANHN, TorchANHN),
        (AttNHP, TorchAttNHP),
        (BaseModel, TorchBaseModel),
        (FullyNN, TorchFullyNN),
        (IntensityFree, TorchIntensityFree),
        (NHP, TorchNHP),
        (ODETPP, TorchODETPP),
        (RMTPP, TorchRMTPP),
        (S2P2, TorchS2P2),
        (SAHP, TorchSAHP),
        (THP, TorchTHP),
        (WSMTHP, TorchWSMTHP),
    )

    assert all(model is torch_alias for model, torch_alias in alias_pairs)
    assert all((EventSampler, LogNormalMixtureDistribution))


def test_model_wrapper_imports_from_flat_module():
    from easy_tpp.model_wrapper import ModelWrapper, TorchModelWrapper

    assert ModelWrapper is TorchModelWrapper
