===================================
``EasyTPP`` Documentation
===================================


``EasyTPP`` is an easy-to-use development and application toolkit for `Neural Temporal Point Process <https://mathworld.wolfram.com/TemporalPointProcess.html>`_ (*Neural TPP*), with key features in configurability, compatibility and reproducibility. We hope this project could benefit both researchers and practitioners with the goal of easily customized development and open benchmarking.


Tutorial Notebooks
==================

The repository contains five walkthroughs:

* `Dataset loading and inspection <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/notebooks/easytpp_1_dataset.ipynb>`_
* `TensorBoard and Weights & Biases <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/notebooks/easytpp_2_tfb_wb.ipynb>`_
* `Model training and evaluation <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/notebooks/easytpp_3_train_eval.ipynb>`_
* `IntensityFree log-space statistics deep dive <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/notebooks/easytpp_intensityfree_log_stats.ipynb>`_
* `Multi-step generation deep dive <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/notebooks/easytpp_multistep_generation.ipynb>`_



.. toctree::
   :hidden:

.. toctree::
   :maxdepth: 2
   :caption: GETTING STARTED

   Introduction <get_started/introduction.rst>
   Installation <get_started/install.rst>
   Quick Start <get_started/quick_start.rst>


.. toctree::
   :maxdepth: 2
   :caption: USER GUIDE

   Dataset <user_guide/dataset.rst>
   Model Training <user_guide/run_train_pipeline.rst>
   Model Prediction <user_guide/run_eval.rst>

.. toctree::
   :maxdepth: 2
   :caption: DEVELOPER GUIDE

   Model Customization <dev_guide/model_custom.rst>


.. toctree::
   :maxdepth: 2
   :caption: ADVANCED TOPICS

   Thinning Algorithm <advanced/thinning_algo.rst>
   Tensorboard <advanced/tensorboard.rst>
   Performance Benchmarks <advanced/performance_valid.rst>
   Implementation Details <advanced/implementation.rst>

.. toctree::
   :maxdepth: 2
   :caption: API REFERENCE

    Config  <ref/config.rst>
    Preprocess  <ref/preprocess.rst>
    Model  <ref/models.rst>
    Runner  <ref/runner.rst>
    Hyper-parameter Optimization  <ref/hpo.rst>
    Torch Wrapper  <ref/wrapper.rst>
    Utilities  <ref/utils.rst>
