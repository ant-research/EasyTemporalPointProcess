==================
Introduction
==================


``EasyTPP`` provides an open-source library for `Neural TPP`, with a fully automated pipeline for model training and prediction.


Framework
=========


``EasyTPP`` is a PyTorch library. Its data processing, model training,
evaluation, and generation pipeline all use the torch backend.


At the module level, ``EasyTPP`` is a package that consists of the following components, which are designed as loose-coupled modules that provide flexibility for users to develop customized functionalities.



========================  ==============================================================================
Name                      Description
========================  ==============================================================================
`Preprocess` module       Provides data batch-wise padding, inter-time processing and other related work for raw sequence.

`Model` module            Implements a list of SOTA TPP models. Please refer to `Model Validation <../advanced/performance_valid.html>`_ for more details.

`Config` module           Encapsulate the construction of the configuration needed to run the pipeline.

`Runner` module           Controls the training and prediction pipeline.
========================  ==============================================================================



Install
=========

``EasyTPP`` can be installed either from PyPI or from source. PyTorch and the
other runtime dependencies are installed with the package.

Please see `Installation <./install.html>`_ for details of requirement and installation.


Prepare Data
============

EasyTPP accepts the legacy pickle representation and JSON files (including
datasets hosted by the `EasyTPP Hugging Face organization
<https://huggingface.co/easytpp>`_). Both representations provide
``time_since_start``, ``time_since_last_event``, ``type_event``, and
``dim_process``. The `Preprocess <../ref/preprocess.html>`_ module turns these
fields into padded model batches.


An example of building a pseudo dataloader can be found in
`examples/data_loader.py <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/data_loader.py>`_.
See `Dataset <../user_guide/dataset.html>`_ for the supported input formats and
the ``TPPDataLoader`` pipeline.


Model Training and Prediction
==============================

The training and prediction pipeline consists of two steps:

1. Setup the config file, which specifies the dataset dir, model params and pipeline settings.
2. Launch the python script to run the whole pipeline.

Please see `Training Pipeline <../user_guide/run_train_pipeline.html>`_ and `Evaluation Pipeline <../user_guide/run_eval.html>`_ for more details.


Implemented Models
==================

Version 0.2.4 provides the following registered model IDs. Their implementation
classes are exported from ``easy_tpp.model`` under the plain model names.
``BaseModel`` is the common implementation base; the other eleven entries are
trainable reference models. The former ``Torch``-prefixed names remain aliases
for backward compatibility.

.. list-table::
   :header-rows: 1
   :widths: 18 62 20

   * - Model
     - Paper title
     - Venue / year
   * - ``ANHN``
     - `Attentive Neural Hawkes Network <https://arxiv.org/abs/2211.11758>`_
     - IJCNN 2021
   * - ``AttNHP``
     - `Transformer Embeddings of Irregularly Spaced Events and Their Participants <https://arxiv.org/abs/2201.00044>`_
     - ICLR 2022
   * - ``FullyNN``
     - `Fully Neural Network based Model for General Temporal Point Processes <https://arxiv.org/abs/1905.09690>`_
     - NeurIPS 2019
   * - ``IntensityFree``
     - `Intensity-Free Learning of Temporal Point Processes <https://arxiv.org/abs/1909.12127>`_
     - ICLR 2020
   * - ``NHP``
     - `The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process <https://arxiv.org/abs/1612.09328>`_
     - NeurIPS 2017
   * - ``ODETPP``
     - `Neural Spatio-Temporal Point Processes <https://arxiv.org/abs/2011.04583>`_ (simplified implementation)
     - ICLR 2021
   * - ``RMTPP``
     - `Recurrent Marked Temporal Point Processes: Embedding Event History to Vector <https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf>`_
     - KDD 2016
   * - ``S2P2``
     - `Deep Continuous-Time State-Space Models for Marked Event Sequences <https://openreview.net/pdf?id=74SvE2GZwW>`_
     - NeurIPS 2025
   * - ``SAHP``
     - `Self-Attentive Hawkes Process <https://arxiv.org/abs/1907.07561>`_
     - ICML 2020
   * - ``THP``
     - `Transformer Hawkes Process <https://arxiv.org/abs/2002.09291>`_
     - ICML 2020
   * - ``WSMTHP``
     - `Is Score Matching Suitable for Estimating Point Processes? <https://arxiv.org/abs/2410.04037>`_
     - NeurIPS 2024
   * - ``BaseModel``
     - Common base class (not a paper model)
     - --
