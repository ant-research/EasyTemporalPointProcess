====================
Quick Start
====================


We use the [Taxi]_ dataset as an example to show how to use ``EasyTPP`` to train a model. More details and results are provided in `Training Pipeline <../user_guide/run_train_pipeline.html>`_.


Download Dataset
===================

The Taxi dataset we used is preprocessed by `HYPRO <https://github.com/iLampard/hypro_tpp>`_ . You can download this dataset `here <https://drive.google.com/drive/folders/1vNX2gFuGfhoh-vngoebaQlj2-ZIZMiBo>`_.


Create the dir to save the pkl files.


Setup the configuration file
==============================

We provide a preset config file in `Example Config <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml>`_. The details of the configuration can be found in `Training Pipeline <../user_guide/run_train_pipeline.html>`_.


Train the Model
=========================

At this stage we need to write a script to run the training pipeline. There is a preset script `train_nhp.py <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/train_nhp.py>`_ and one can simply copy it.

After the setup of data, config and running script, the directory structure is as follows:

.. code-block:: bash

    data
     |______taxi
             |____ train.pkl
             |____ dev.pkl
             |____ test.pkl

    configs
     |______experiment_config.yaml

     train_nhp.py



The one can simply run the following command.


.. code-block:: bash

    python train_nhp.py



Reference
----------

.. [Taxi]

.. code-block:: bash

    @misc{whong-14-taxi,
      title = {F{OIL}ing {NYC}’s Taxi Trip Data},
      author={Whong, Chris},
      year = {2014},
      url = {https://chriswhong.com/open-data/foil_nyc_taxi/}
    }

