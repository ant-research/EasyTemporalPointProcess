==================
Installation
==================


``EasyTPP`` provides an open-source library for `Neural TPP`, with a fully automated pipeline for model training and prediction.


Requirements
=============

EasyTPP 0.2.4 requires Python 3.9 or newer and uses PyTorch as its only model
backend. Create an isolated Python environment before installing the package.
For example:

.. code-block:: bash

    conda create -n easytpp python=3.11
    conda activate easytpp



Install
=====================


Install with pip
--------------------------


.. code-block:: bash

    pip install easy-tpp


Install with the source
--------------------------

Setup from the source：

.. code-block:: bash

    git clone https://github.com/ant-research/EasyTemporalPointProcess.git
    cd EasyTemporalPointProcess
    pip install -e .
