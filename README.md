# Deep-Mutual-Learning
Implementation of Deep Mutual Learning by Pytorch to do classification on cifar100.  
The algorithm was proposed in *《Deep Mutual Learning》 (CVPR 2017)*.
# Dependence
Pytorch 1.0.0  
tensorboard 1.14.0
# Overview
Overview of the algorithm:  
<img src="https://raw.githubusercontent.com/chxy95/Deep-Mutual-Learning/master/images/Overview.png" width="700"/>
# Usage
The default network for DML is ResNet32.  
Train 2 models using DML by main.py:  
```
python train.py --model_num 2
```
Use tensorboard to monitor training process on choosing port:
```
tensorboard --logdir logs --port 6006
```
# Result
| Network | ind_avg_acc | Dml_avg_acc|
|---------|------------:|-----------:|
|ResNet32 |   69.83%    | **71.03%** |
