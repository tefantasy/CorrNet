# CorrNet

This project implements CorrNet, a video action recognition network proposed in paper [Video Modeling with Correlation Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Video_Modeling_With_Correlation_Networks_CVPR_2020_paper.pdf). 

## Install
Before running experiment scripts, please first install this project in development mode by

``pip3 install -e <path_to_project>``

## Data preparation, training and testing

Please refer to the [PySlowFast](https://github.com/facebookresearch/SlowFast) repository since a large proportion of this project directly adopts the code from PySlowFast. 

## About this repository

`corr_net/models/correlation_net` contains code for the CorrNet model. 

Please note that weighted correlation layer is implemented using pure PyTorch API, which consumes relatively high memory. 

## Experiment results

| architecture | depth | top-1 acc | top-5 acc | 
|----------|-------|-------|-------|
| corr_26 | R26 | 76.46 | 93.05 |
| corr_50 | R50 | 77.58 | 93.34 |

All the experiments are conducted on Kinetics-600 (we collected ~368k training examples) dataset. We sample 16 frames as input to the model, with sampling rate of 8. We use cosine decay learning rate policy with warmup: base_lr=0.2, warmup_epochs=30, num_epochs=350. 


## CUDA extension for correlation layer

Please refer to [this repository](https://github.com/tefantasy/WeightedCorrelationExtension). Class `WeightedCorrelationLayer` in `corr_net/models/correlation_net/corr.py` can be directly replaced by class `WeightedCorrelationLayerExtension` in the CUDA implementation. 

## Acknowledgement

**[PySlowFast](https://github.com/facebookresearch/SlowFast)**
```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

**[R2Plus1D-PyTorch](https://github.com/irhum/R2Plus1D-PyTorch)**

## References

**[Video Modeling with Correlation Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Video_Modeling_With_Correlation_Networks_CVPR_2020_paper.pdf)**
```BibTeX
@inproceedings{Wang_2020_CVPR,
author = {Wang, Heng and Tran, Du and Torresani, Lorenzo and Feiszli, Matt},
title = {Video Modeling With Correlation Networks},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
