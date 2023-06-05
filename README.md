# EasyTPP

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![](https://img.shields.io/badge/license-Apache-000000.svg)
![GitHub last commit]([https://img.shields.io/github/last-commit/enze5088/Chatterbox](https://img.shields.io/github/last-commit/ant-research/EasyTemporalPointProcess))

** We are actively updating the repo now. We will try to finish the first version in a few days. **

`EasyTPP` is an easy-to-use development and application toolkit for [Temporal Point Process](https://mathworld.wolfram.com/TemporalPointProcess.html) (TPP), with key features in configurability, compatibility and reproducibility. We hope this project could benefit both researchers and practitioners with the goal of easily customized development and open benchmarking in TPP.
<span id='top'/>



| <a href='#features'>Features</a>|<a href='#requirement'>Install</a>  | <a href='#model-list'>Model List</a> | <a href='#dataset'>Dataset</a>  | <a href='#quick-start'>Quick Start</a> | <a href='#doc'>Documentation</a> | <a href='#citation'>Citation</a> |<a href='#acknowledgment'>Acknowledgement</a> | <a href='#star-history'>Star History</a> | 

## News
<span id='news'/>

- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [05-29-2023] We release the paper [Language Model Can Improve Event Prediction by Few-Shot Abductive Reasoning](https://arxiv.org/abs/2305.16646), in which we propose a LLM-guided event prediction model!
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [05-29-2023] We release ``EasyTPP`` v0.1.0!
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


## Dataset <a href='#top'>[Back to Top]</a>
<span id='dataset'/>

We preprocessed one synthetic and five real world datasets from widely-cited works that contain diverse characteristics in terms of their application domains and temporal statistics:
- Synthetic: a univariate Hawkes process simulated by [Tick](https://github.com/X-DataInitiative/tick) library.
- Retweet ([Zhou, 2013](http://proceedings.mlr.press/v28/zhou13.pdf)): timestamped user retweet events.
- Taxi ([Whong, 2014](https://chriswhong.com/open-data/foil_nyc_taxi/)): timestamped taxi pick-up events.
- StackOverflow ([Leskovec, 2014](https://snap.stanford.edu/data/)): timestamped user badge reward events in StackOverflow.
- Taobao ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): timestamped user online shopping behavior events in Taobao platform.
- Amazon ([Amazon Review, 2018](https://nijianmo.github.io/amazon/)): timestamped user online shopping behavior events in Amazon platform.

  All datasets are preprocess to the `Gatech` format dataset widely used for TPP researchers, and saved at [Google Drive](https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7) with a public access.



## License <a href='#top'>[Back to Top]</a>

This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/EasyNLP/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/ant-research/EasyTPP/blob/master/NOTICE) file for more information.


## Citation <a href='#top'>[Back to Top]</a>

<span id='citation'/>

If you find `EasyTPP` useful for your research or development, please cite the following <a href="https://arxiv.org/abs/2204.05011" target="_blank">paper</a>:
```
comming soon
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

