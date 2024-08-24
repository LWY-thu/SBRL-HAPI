# SBRL-HAPI

## Introduction

This repository contains the implementation code for paper:

**Estimating Heterogeneous Treatment Effect with Distribution Shift and Unmeasured Confounders**

Wenyang Liu, Yuling Zhang, Anpeng Wu, Kun Kuang, Ming Ma, Zhi Wang

As an extension of our prior conference paper [1] published in IEEE ICDE2024, this paper broaden the problem scope to a more challenging scenario where we may not fully observe all the confounders, and extend the preceding SBRL-HAP framework with an Instrumental Regularizer to address this issue.


## Requirements

Python 3.6.8 with TensorFlow 1.15.0, NumPy 1.19.5, Scikit-learn 0.24.2 and MatplotLib 3.3.4.

## Instructions

`generator/dataGenerator.py` is an example of Synthetic Data Generation.

`model/sbrl_hapi.py` contains the class for SBRL-HAPI, which is implemented on the network backbone of the Counterfactual Regression [2].

`util/utils.py` includes the necessary utilities.

Run `train.py` scripts to train the model.

```python
python train.py
```

## Reference
[1] Zhang, Yuling, et al. "Stable heterogeneous treatment effect estimation across out-of-distribution populations." 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 2024.
[2] Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment effect: generalization bounds and algorithms." International conference on machine learning. PMLR, 2017.
