====================
Quick Start
====================


We use the [Taxi]_ dataset as an example to show how to use ``EasyTPP`` to train a model. More details and results are provided in `Training Pipeline <../user_guide/run_train_pipeline.html>`_.


Download Dataset
===================



The Taxi dataset we used is preprocessed by `HYPRO <https://github.com/iLampard/hypro_tpp>`_ . You can either download the dataset (in pickle) from Google Drive `here <https://drive.google.com/drive/folders/1vNX2gFuGfhoh-vngoebaQlj2-ZIZMiBo>`_ or the dataset (in json) from `HuggingFace <https://huggingface.co/easytpp>`_.


Note that if the data sources are pickle files, we need to write the data config (in `Example Config <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml>`_) in the following way

.. code-block:: yaml

    data:
      taxi:
        data_format: pickle
        train_dir: ./data/taxi/train.pkl
        valid_dir: ./data/taxi/dev.pkl
        test_dir: ./data/taxi/test.pkl

If we choose to directly load from HuggingFace, we can put it this way:

.. code-block:: yaml

    data:
      taxi:
        data_format: json
        train_dir: easytpp/taxi
        valid_dir: easytpp/taxi
        test_dir: easytpp/taxi


Meanwhile, it is also feasible to put the local directory of json files downloaded from HuggingFace in the config:

.. code-block:: yaml

    data:
      taxi:
        data_format: json
        train_dir: ./data/taxi/train.json
        valid_dir: ./data/taxi/dev.json
        test_dir: ./data/taxi/test.json




Setup the configuration file
==============================

We provide a preset config file in `Example Config <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml>`_. The details of the configuration can be found in `Training Pipeline <../user_guide/run_train_pipeline.html>`_.




Train the Model
=========================

At this stage we need to write a script to run the training pipeline. There is a preset script `train_nhp.py <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/train_nhp.py>`_ and one can simply copy it.

Taking the pickle data source for example, after the setup of data, config and running script, the directory structure is as follows:

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

