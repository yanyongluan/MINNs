
# Revisiting Multiple Instance Neural Networks

By [Xinggang Wang](http://mclab.eic.hust.edu.cn/~xwang/index.htm), Yongluan Yan, [Peng Tang](https://ppengtang.github.io/), [Xiang Bai](http://mclab.eic.hust.edu.cn/~xbai/), and [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu).

### Introduction

**Multiple Instance Neural Networks(MINNs)** are neural networks that aim at solving the MIL problems in an end-to-end manner.

- It is centered on learning bag representation in the nueral network. And recent deep learning tricks including deep supervision, and residual connections are studied in MINNs.
- The proposed MINNs achieve state-of-the-art or competitive performance on several MIL benchmarks. Moreover, it is extremely fast for both testing and training, for example, it takes only 0.0003 seconds to predict a bag and a few seconds to train on MIL datasets on a moderate CPU.
- Our code is written by Python, based on [keras](https://keras.io/), which use [Theano](http://deeplearning.net/software/theano/) as backend.

The paper has been accepted by Pattern Recognition, 2017. For more details, please refer to our [paper](http://www.sciencedirect.com/science/article/pii/S0031320317303382).

### Citing MINNs

If you find MINNs useful in your research, please consider citing:

    @article{wang2016revisiting,
            title={Revisiting Multiple Instance Neural Networks},
            author={Wang, Xinggang and Yan, Yongluan and Tang, Peng and Bai, Xiang and Liu, Wenyu},
            journal={arXiv preprint arXiv:1610.02501},
            year={2016}
    }


### Requirements: software

1. Requirements for `Keras` and `Theano` (see: [Keras installation instructions](https://keras.io/#installation))

  **Note:** The version of Keras is 1.1.0.

2. Python packages you might not have: `numpy`, `scipy`, and `sklearn`

### Requirements: hardware

1. moderate CPU

### Experimental hyper-parameters

| dataset        | start learning rate   |  weight decay  |  momentum |
| :------:   | :----:  | :----:  |  :----:  |
| musk1      | 0.0005  |  0.005  |  0.9     |
| musk2      | 0.0005  |  0.03   |  0.9     |
| fox        | 0.0001  |  0.01   |  0.9     |
| tiger      | 0.0005  |  0.005  |  0.9     |
| elephat    | 0.0001  |  0.005  |  0.9     |
| 20 newsgroups |  0.001 |  0.001 | 0.9     |
In addition, the number of max epoch is set to 50.

### Usage

**Download dataset**

First download and extract all dataset [Musk](http://www.miproblems.org/datasets/musk/), [animal](http://www.miproblems.org/datasets/foxtigerelephant/), and [Newsgroups](http://www.miproblems.org/datasets/birds-2/) to one directory named `dataset`

### Train
There exist four Python code `mi_Net.py`, `MI_Net.py`, `MI_Net_with_DS.py`, and `MI_Net_with_RC.py` corresponding to four MINN method mentioned in our paper.
We have given default params on MUSK1 dataset as example.

```Shell
    #  run MI-Net on MUSK1 dataset
    python MI_Net.py --dataset musk1 --pooling max --lr 5e-4 --decay 0.005 --momentum 0.9 --epoch 50
```
