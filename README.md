# EasyTPP

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![](https://img.shields.io/badge/license-Apache-000000.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![GitHub last commit](https://img.shields.io/github/last-commit/ant-research/EasyTemporalPointProcess)

![Stars](https://img.shields.io/github/stars/ant-research/EasyTemporalPointProcess)
[![PyPI version](https://img.shields.io/pypi/v/easy-tpp.svg?style=flat-square&color=b7534)](https://pypi.python.org/pypi/easy-tpp/)
[![Downloads](https://static.pepy.tech/personalized-badge/easy-tpp?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/easy-tpp)
![Visitor Badge](https://visitor-badge.feriirawann.repl.co/?username=iLampard&repo=ant-research/EasyTemporalPointProcess&contentType=svg)


`EasyTPP` is an easy-to-use development and application toolkit for [Temporal Point Process](https://mathworld.wolfram.com/TemporalPointProcess.html) (TPP), with key features in configurability, compatibility and reproducibility. We hope this project could benefit both researchers and practitioners with the goal of easily customized development and open benchmarking in TPP.
<span id='top'/>



| <a href='#features'>Features</a>  | <a href='#model-list'>Model List</a> | <a href='#dataset'>Dataset</a>  | <a href='#quick-start'>Quick Start</a> | <a href='#benchmark'>Benchmark</a> |<a href='#doc'>Documentation</a> |<a href='#todo'>Todo List</a> | <a href='#citation'>Citation</a> |<a href='#acknowledgment'>Acknowledgement</a> | <a href='#star-history'>Star History</a> | 

## News
<span id='news'/>

- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [09-30-2023] We published two textual event sequence datasets [GDELT](https://drive.google.com/drive/folders/1Ms-ATMMFf6v4eesfJndyuPLGtX58fCnk) and [Amazon-text-review](https://drive.google.com/drive/folders/1-SLYyrl7ucEG7NpSIF0eSoG9zcbZagZw) that are used in our paper [LAMP](https://arxiv.org/abs/2305.16646), where LLM can be applied for event prediction! See [Documentation](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/dataset.html#preprocessed-datasets) for more details.
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [09-30-2023] Two of our papers [Language Model Can Improve Event Prediction by Few-Shot Abductive Reasoning](https://arxiv.org/abs/2305.16646) (LAMP) and [Prompt-augmented Temporal Point Process for Streaming Event Sequence](https://arxiv.org/abs/2310.04993) (PromptTPP) are accepted by NeurIPS'2023!
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [09-02-2023] We published two non-anthropogenic datasets [earthquake](https://drive.google.com/drive/folders/1ubeIz_CCNjHyuu6-XXD0T-gdOLm12rf4) and [volcano eruption](https://drive.google.com/drive/folders/1KSWbNi8LUwC-dxz1T5sOnd9zwAot95Tp?usp=drive_link)! See <a href='#dataset'>Dataset</a> for details.
- [05-29-2023] We released ``EasyTPP`` v0.0.1!
- [12-27-2022] Our paper [Bellman Meets Hawkes: Model-Based Reinforcement Learning via Temporal Point Processes](https://arxiv.org/abs/2201.12569) was accepted by AAAI'2023!
- [10-01-2022] Our paper [HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences](https://arxiv.org/abs/2210.01753) was accepted by NeurIPS'2022!
- [05-01-2022] We started to develop `EasyTPP`.


## Features <a href='#top'>[Back to Top]</a>
<span id='features'/>

- **Configurable and customizable**: models are modularized and configurable，with abstract classes to support developing customized
  TPP models.
- **Compatible with both Tensorflow and PyTorch framework**: `EasyTPP` implements two equivalent sets of models, which can
  be run under Tensorflow (both Tensorflow 1.13.1 and Tensorflow 2.0) and PyTorch 1.7.0+ respectively. While the PyTorch models are more popular among researchers, the compatibility with Tensorflow is important for industrial practitioners.
- **Reproducible**: all the benchmarks can be easily reproduced.
- **Hyper-parameter optimization**: a pipeline of [optuna](https://github.com/optuna/optuna)-based HPO is provided.


## Model List <a href='#top'>[Back to Top]</a>
<span id='model-list'/>

We provide reference implementations of various state-of-the-art TPP papers:

| No  | Publication |     Model     | Paper                                                                                                                                    | Implementation                                                                                                             |
|:---:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
|  1  |   KDD'16    |     RMTPP     | [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) | [Tensorflow](easy_tpp/model/tf_model/tf_rmtpp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_rmtpp.py)                   |
|  2  | NeurIPS'17  |      NHP      | [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)                     | [Tensorflow](easy_tpp/model/tf_model/tf_nhp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_nhp.py)                       |
|  3  | NeurIPS'19  |    FullyNN    | [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690)                                | [Tensorflow](easy_tpp/model/tf_model/tf_fullnn.py)<br/>[Torch](easy_tpp/model/torch_model/torch_fullynn.py)                |
|  4  |   ICML'20   |     SAHP      | [Self-Attentive Hawkes process](https://arxiv.org/abs/1907.07561)                                                                        | [Tensorflow](easy_tpp/model/tf_model/tf_sahp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_sahp.py)                     |
|  5  |   ICML'20   |      THP      | [Transformer Hawkes process](https://arxiv.org/abs/2002.09291)                                                                           | [Tensorflow](easy_tpp/model/tf_model/tf_thp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_thp.py)                       |
|  6  |   ICLR'20   | IntensityFree | [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)                                                  | [Tensorflow](easy_tpp/model/tf_model/tf_intensity_free.py)<br/>[Torch](easy_tpp/model/torch_model/torch_intensity_free.py) |
|  7  |   ICLR'21   |    ODETPP     | [Neural Spatio-Temporal Point Processes (simplified)](https://arxiv.org/abs/2011.04583)                                                  | [Tensorflow](easy_tpp/model/tf_model/tf_ode_tpp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_ode_tpp.py)               |
|  8  |   ICLR'22   |    AttNHP     | [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://arxiv.org/abs/2201.00044)                           | [Tensorflow](easy_tpp/model/tf_model/tf_attnhp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_attnhp.py)                 |



## Dataset <a href='#top'>[Back to Top]</a>
<span id='dataset'/>

We preprocessed one synthetic and five real world datasets from widely-cited works that contain diverse characteristics in terms of their application domains and temporal statistics:
- Synthetic: a univariate Hawkes process simulated by [Tick](https://github.com/X-DataInitiative/tick) library.
- Retweet ([Zhou, 2013](http://proceedings.mlr.press/v28/zhou13.pdf)): timestamped user retweet events.
- Taxi ([Whong, 2014](https://chriswhong.com/open-data/foil_nyc_taxi/)): timestamped taxi pick-up events.
- StackOverflow ([Leskovec, 2014](https://snap.stanford.edu/data/)): timestamped user badge reward events in StackOverflow.
- Taobao ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): timestamped user online shopping behavior events in Taobao platform.
- Amazon ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): timestamped user online shopping behavior events in Amazon platform.

Per users' request, we processed two non-anthropogenic datasets 
- [Earthquake](https://drive.google.com/drive/folders/1ubeIz_CCNjHyuu6-XXD0T-gdOLm12rf4): timestamped earthquake events over the Conterminous U.S from 1996 to 2023, processed from [USGS](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data).
- [Volcano eruption](https://drive.google.com/drive/folders/1KSWbNi8LUwC-dxz1T5sOnd9zwAot95Tp?usp=drive_link): timestamped volcano eruption events over the world in recent hundreds of years, processed from [The Smithsonian Institution](https://volcano.si.edu/).


  All datasets are preprocess to the `Gatech` format dataset widely used for TPP researchers, and saved at [Google Drive](https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7) with a public access.

## Quick Start <a href='#top'>[Back to Top]</a>
<span id='quick-start'/>

We provide an end-to-end example for users to run a standard TPP model with `EasyTPP`.


### Step 1. Installation

First of all, we can install the package either by using pip or from the source code on Github.

To install the latest stable version:
```bash
pip install easy-tpp
```

To install the latest on GitHub:
```bash
git clone https://github.com/ant-research/EasyTemporalPointProcess.git
cd EasyTemporalPointProcess
python setup.py install
```


### Step 2. Prepare datasets 

We need to put the datasets in a local directory before running a model and the datasets should follow a certain format. See [OnlineDoc - Datasets](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/dataset.html) for more details.

Suppose we use the [taxi dataset](https://chriswhong.com/open-data/foil_nyc_taxi/) in the example.

### Step 3. Train the model


Before start training, we need to set up the config file for the pipeline. We provide a preset config file in [Example Config](https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml). The details of the configuration can be found in [OnlineDoc - Training Pipeline](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/run_train_pipeline.html).

After the setup of data and config, the directory structure is as follows:

```bash

    data
     |______taxi
             |____ train.pkl
             |____ dev.pkl
             |____ test.pkl

    configs
     |______experiment_config.yaml

```


Then we start the training by simply running the script 

```python

import argparse
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='NHP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

    model_runner = Runner.build_from_config(config)

    model_runner.run()


if __name__ == '__main__':
    main()

```

A more detailed example can be found at [OnlineDoc - QuickStart](https://ant-research.github.io/EasyTemporalPointProcess/get_started/quick_start.html).


## Documentation <a href='#top'>[Back to Top]</a>
<span id='doc'/>

The classes and methods of `EasyTPP` have been well documented so that users can generate the documentation by:

```shell
cd doc
pip install -r requirements.txt
make html
```
NOTE:
* The `doc/requirements.txt` is only for documentation by Sphinx, which can be automatically generated by Github actions `.github/workflows/docs.yml`. (Trigger by pull request.)

The full documentation is available on the [website](https://ant-research.github.io/EasyTemporalPointProcess/).
 
## Benchmark <a href='#top'>[Back to Top]</a>
<span id='benchmark'/>

In the [examples](https://github.com/ant-research/EasyTemporalPointProcess/tree/main/examples) folder, we provide a [script](https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/benchmark_script.py) to benchmark the TPPs, with Taxi dataset as the input. 

To run the script, one should download the Taxi data following the above instructions. The [config](https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml) file is readily setup up. Then run


```shell
cd examples
python benchmark_script.py
```


## License <a href='#top'>[Back to Top]</a>

This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/EasyNLP/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/ant-research/EasyTPP/blob/master/NOTICE) file for more information.


## Todo List <a href='#top'>[Back to Top]</a>
<span id='todo'/>

- [x] New dataset:
  - [x] Earthquake: the source data is available in [USGS](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data).
  - [x] Volcano eruption: the source data is available in [NCEI](https://www.ngdc.noaa.gov/hazard/volcano.shtml).
- [ ] New model:
  - [ ] Meta Temporal Point Process, ICLR 2023.
  - [ ] Model-based RL via TPP, AAAI 2022. 

## Citation <a href='#top'>[Back to Top]</a>

<span id='citation'/>

If you find `EasyTPP` useful for your research or development, please cite the following <a href="https://arxiv.org/abs/2204.05011" target="_blank">paper</a>:
```
@article{xue2023easytpp,
      title={EasyTPP: Towards Open Benchmarking the Temporal Point Processes}, 
      author={Siqiao Xue and Xiaoming Shi and Zhixuan Chu and Yan Wang and Fan Zhou and Hongyan Hao and Caigao Jiang and Chen Pan and Yi Xu and James Y. Zhang and Qingsong Wen and Jun Zhou and Hongyuan Mei},
      year={2023},
      eprint={2307.08097},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgment <a href='#top'>[Back to Top]</a>
<span id='acknowledgment'/>

The project is jointly initiated by Machine Intelligence Group, Alipay and DAMO Academy, Alibaba. 

The following repositories are used in `EasyTPP`, either in close to original form or as an inspiration:

- [EasyRec](https://github.com/alibaba/EasyRec)
- [EasyNLP](https://github.com/alibaba/EasyNLP)
- [FuxiCTR](https://github.com/xue-pai/FuxiCTR)
- [Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes)
- [Neural Hawkes Particle Smoothing](https://github.com/hongyuanmei/neural-hawkes-particle-smoothing)
- [Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)
- [Huggingface - transformers](https://github.com/huggingface/transformers)


## Star History <a href='#top'>[Back to Top]</a>
<span id='star-history'/>

![Star History Chart](https://api.star-history.com/svg?repos=ant-research/EasyTemporalPointProcess&type=Date)

