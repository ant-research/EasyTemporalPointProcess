==================
Introduction
==================


``EasyTPP`` provides an open-source library for `Neural TPP`, with a fully automated pipeline for model training and prediction.


Framework
=========


``EasyTPP`` supports both Tensorflow and PyTorch: each model has two equivalent versions implemented in Tensorflow 1.13 and Pytorch 1.8 respectively. The data processing and model training / prediction pipeline are compatible with both Tensorflow and Pytorch as well.


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

``EasyTPP`` can be installed either by pip or the source. By default it is built based on PyTorch. If one wants to run with the Tensorflow backend, one needs to install Tensorflow additionally.

Please see `Installation <./install.html>`_ for details of requirement and installation.


Prepare Data
============

By default, we use the data in Gatech format, i.e., each dataset is a dict containing the keys such as `time_since_last_event`, `time_since_start` and `type_event`. `Preprocess <../ref/preprocess.html>`_ module
will preprocess the data and feed it into the model.


An example of building a pseudo dataloader can be found at `examples <https://github.com/ant-research/EasyTemporalPointProcess/tree/main/examples/data_loader.py>`_. Please refer to `Datatset <../user_guide/dataset.html>`_ for more explanations of the `TPP` dataset iterator.


Model Training and Prediction
==============================

The training and prediction pipeline consists of two steps:

1. Setup the config file, which specifies the dataset dir, model params and pipeline settings.
2. Launch the python script to run the whole pipeline.

Please see `Training Pipeline <../user_guide/run_train_pipeline.html>`_ and `Evaluation Pipeline <../user_guide/run_eval.html>`_ for more details.