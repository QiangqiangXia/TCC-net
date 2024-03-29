# TCC-net
This repository is an official implementation of the paper [TCC-net: A two-stage training method with contradictory loss and co-teaching based on meta-learning for learning with noisy labels](https://www.sciencedirect.com/science/article/pii/S0020025523005935?utm_campaign=STMJ_AUTH_SERV_PUBLISHED&utm_medium=email&utm_acid=268358976&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED&CMX_ID=&utm_in=DM365873&utm_source=AC_).

If you find TCC-net useful in your research then please cite:

`Xia Q, Lee F, Chen Q. TCC-net: A two-stage training method with contradictory loss and co-teaching based on meta-learning for learning with noisy labels[J]. Information Sciences, 2023: 639, 119008.`

With the rapid development of deep neural networks, the demand for large-scale accurately labeled datasets is growing rapidly. However, human-labeled datasets often produce misleading information due to mistakes in manual labeling. Most previous works fail to control well the overfitting of the trained model to noisy labels. In this paper, we propose a novel two-stage learning framework for learning with noisy labels, called the Two-stage training method with Contradictory loss and Co-teaching based on meta-learning (TCC-net). First, a novel robust loss function called contradictory loss is designed for pre-training, which is proved to be sufficiently robust both in the experimental results and theoretical foundation. During the training stage, we co-teach two pretrained networks based on meta-learning without any auxiliary clean subset as meta-data. Unlike other co-teaching methods, we introduce two multilayer perceptron to assist in weighting the selected samples, meaning each network updates itself with weighted-selected samples by its peer network and self-perceptron. Experimental results on corrupted datasets, such as Cifar10, Cifar100, Animal10N, and Clothing1M, demonstrate that TCC-net is superior to other state-of-the-art methods on shallower layers. Specifically, we achieve 12.59% improvement on synthetic Cifar10 with 80% symmetric noise and 0.27% on the real-world Animal10N dataset.

<div align=center>
<img src="https://github.com/QiangqiangXia/TCC-net/blob/main/TCC-net.png"> <width="500" height="300">
</div>

# Setups

The environment is as bellow:

* Python
* Pytorch
* Tensorboard

# Running TCC-net on benchmark datasets (Cifar10, Cifar100, Animal10N and Clothing1M)

Here is an example:

```python
python main.py --dataset cifar10 --noise_type symmetric --percent 0.2
```

# Reference

[1]: Thank for [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://github.com/bhanML/Co-teaching).
