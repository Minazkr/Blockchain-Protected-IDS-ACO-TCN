# Raspberry Pi deployment
This directory contains optimised native C++ code for running the TCN anomaly detection model on resource-constrained hardware, such as the Raspberry Pi Zero W. The implementation is designed to run without heavy machine learning libraries, ensuring minimal memory consumption and high processing speed (inference).

## Contents:
ids_pi.cpp: The primary lightweight C++ inference script. It reads network flow data in CSV format, handles scaling and windowing, executes the TCN model and outputs the predicted class alongside execution latency. It also includes a self-test feature (--check) to verify consistency with Python results.

export_weights.py: A helper Python script executed on the training machine after training. It extracts preprocessing statistics (median, mean and standard deviation) and model weights and converts them into a header file (model_weights.h) for the C++ code.

## Usage and execution guide:
1. Generate weights and parameters (model_weights.h).
After training the model in Python, run the following command to export the weights and normalisation parameters for C++:
