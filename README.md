# MPKE

Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Citation](#citation)

## Overview

## Requirements

#### Environment:

- Python 3.6
- CUDA 9.0

#### Pacakges
You can click the following links for detailed installation instructions.
- [Pytorch 1.1](https://pytorch.org/get-started/previous-versions/)
- [NumPy 1.16.2](https://www.scipy.org/install.html)

## Quickstart

#### Train and Test

Hyperparameter can be adjusted as follows: For example, if you want to change the number of learning rate to 0.001, you can append `--lr 0.001` after `train.py`
```
python train.py
```
For YAGO11K dataset, the best result can be obtained by the following settings:
```
python train.py --dataset YAGO11K --margin 12 --lr 0.005 --norm 1 --norm_m 2 --hyper_m 0.5 --ns 10
```
For WIKI12K dataset, the best result can be obtained by the following settings:
```
python train.py --dataset WIKI12K --margin 18 --lr 0.005 --norm 1 --norm_m 1 --hyper_m 0.1 --ns 10
```
For WIKI11K dataset, the best result can be obtained by the following settings:
```
python train.py --dataset WIKI11K --margin 18 --lr 0.001 --norm 1 --norm_m 2 --hyper_m 0.5 --ns 3
```

 We train each dataset with default setting as 1000 epochs and with each 20 epochs of training, the code will run the validate dataset and write the result into ./dataset_name/log.txt.
 
 After training, the test of model will be runed auromaticlly by our code and the result will be writen into ./dataset_name/result.txt
