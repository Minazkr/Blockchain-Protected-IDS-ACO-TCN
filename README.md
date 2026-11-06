# Reproducibility Package

This repository contains the main implementation files and experimental results of our proposed intrusion detection framework. It is intended to provide readers with a clear overview of the implementation and the experimental setup used in this study.

# A Blockchain-Protected Lightweight Intrusion Detection Framework Using ACO and TCN for Resource-Constrained IoT Devices


## Overview

This repository provides the implementation of a lightweight intrusion detection framework designed for resource-constrained IoT environments. The method combines Information Gain–guided Ant Colony Optimization (IG-ACO) for feature selection with a causal Temporal Convolutional Network (TCN) for classification. 

The experimental protocol is designed to avoid common sources of leakage. Feature selection, scaling, and categorical encoding are fitted exclusively on the training set. Chronological splits with a purge gap are used for CICIDS2017, and the official train/test files are respected for NSL-KDD. Class imbalance is explicitly addressed, with particular attention given to rare attack classes through class weighting and detailed per-class reporting.

## Method

The framework consists of three main components:

1. **Feature Selection**: Information Gain combined with Ant Colony Optimization (IG-ACO) is used to select a compact set of 8 features.
2. **Classification**: A lightweight causal Temporal Convolutional Network (TCN) processes sequences of network flows. Class imbalance is addressed by assigning higher weights to rare attack classes during training.
3. **Integrity Protection**: Critical detection events are recorded in a permissioned ledger using ECDSA signatures.

## Repository Contents

- Source code for data preprocessing and feature selection
- ACO configuration and selected feature indices
- TCN model implementation and trained weights
- Experimental configurations and evaluation results

## Environment

Experiments were run on **Google Colab** with the following configuration:

- Python: 3.13.16
- Operating system: Ubuntu 24.04.4 LTS
- Runtime: NVIDIA Tesla T4 GPU (15 GB memory), driver 580.82.07, CUDA 13.0
- Package versions: see `environment_freeze.txt`

Latency and energy measurements were performed on a Raspberry Pi, not on Colab. See the Hardware measurements section.

## Datasets

- **CICIDS2017:** (https://www.unb.ca/cic/datasets/index.html)
- **NSL-KDD:** (https://www.unb.ca/cic/datasets/nsl.html)
- **Edge-IIoTset Cyber Security Dataset of IoT & IIoT(https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)

## Reproducibility

This repository provides the main implementation files and experimental configurations used in our study. The included source code, selected feature indices, model weights, and evaluation results allow readers to examine the proposed method and reproduce the experimental setup described in the paper.

## Citation

If you use this code or find it useful, please cite our paper:

```bibtex
@article{YourKey2026,
  title={A Blockchain-Protected Lightweight Intrusion Detection Framework Using ACO and TCN for Resource-Constrained IoT Devices},
  author={Author1 and Author2 and Author3},
  journal={Journal Name},
  year={2026}
}
