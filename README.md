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
