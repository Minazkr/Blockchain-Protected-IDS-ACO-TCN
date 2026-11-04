# Reproducibility Package

This repository contains the main implementation files and experimental results of our proposed intrusion detection framework. It is intended to provide readers with a clear overview of the implementation and the experimental setup used in this study.

## Overview

This repository provides the implementation of a lightweight intrusion detection framework designed for resource-constrained IoT environments. The method combines Information Gain–guided Ant Colony Optimization (IG-ACO) for feature selection with a causal Temporal Convolutional Network (TCN) for classification. A permissioned ledger based on ECDSA is included to protect critical detection events.

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

## Dataset

The experiments were conducted using the **CICIDS2017** dataset.  
Due to size limitations, the original dataset is not included in this repository.  
You can download it from the official source:  
[https://www.unb.ca/cic/datasets/index.html](https://www.unb.ca/cic/datasets/index.html)

## Reproducibility

This repository provides the main implementation files and experimental configurations used in our study. The included source code, selected feature indices, model weights, and evaluation results allow readers to examine the proposed method and reproduce the experimental setup described in the paper.
