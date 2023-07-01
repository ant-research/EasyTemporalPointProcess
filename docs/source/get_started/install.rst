==================
Installation
==================


``EasyTPP`` provides an open-source library for `Neural TPP`, with a fully automated pipeline for model training and prediction.


Requirements
=============

.. code-block:: bash

    PyTorch version >= 1.8.0
    Python version >= 3.7
    Tensorflow version >= 1.13.1 (only needed when using Tensorflow backend)



First, we need a python environment whose version is at least greater than 3.7.0. If you don’t have one, please refer to the `Documentation <https://docs.anaconda.com/anaconda/install/>`_ to install and configure the Anaconda environment.

.. code-block:: bash

    conda create -n easytpp python=3.8
    conda activate easytpp

Then, install Pytorch and keep the version at least greater than 1.8.0.

.. code-block:: bash

    pip install torch

By default, we assume to use PyTorch. If one wants to use Tensorflow backend, please install tensorflow additionally. Both Tensorflow 1.13.1 and 2.x are supported.

.. code-block:: bash

    pip install tensorflow



Install
=====================


Install with pip
--------------------------


.. code-block:: bash

    pip install easy_tpp


Install with the source
--------------------------

Setup from the source：

.. code-block:: bash

    git clone https://github.com/ant-research/EasyTemporalPointProcess.git
    cd EasyTemporalPointProcess
    python setup.py install

