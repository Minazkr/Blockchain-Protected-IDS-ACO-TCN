#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export TCN Model Weights and Scaler Statistics to C++ Header
------------------------------------------------------------
This script runs on the training machine after model training is complete.
- Fits the preprocessing pipeline exclusively on the training split.
- Exports model weights, normalization parameters (mean, std, median), class names, 
  and verification vectors into a single C++ header file (`model_weights.h`).

Usage Example:
    python export_weights.py --train-csv train_split.csv --checkpoint tcn.pt \
        --classes "Normal,Anomalous" --out model_weights.h

Note: --classes must list the class names in the exact same order as the training label encoding.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Selected features (Channel order strictly matches Section 4 of the paper)
FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Destination Port",
    "Flow IAT Mean",
    "Bwd Packet Length Mean",
]
WINDOW = 16
KERNEL = 3
DILATIONS = [1, 2]
CHANNELS = [16, 32]
MAX_MISSING_FRACTION = 0.05


class CausalBlock(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv1 = nn.Conv1d(cin, cout, k, dilation=d)
        self.conv2 = nn.Conv1d(cout, cout, k, dilation=d)
        self.down = nn.Conv1d(cin, cout, 1)  # 1x1 residual projection

    def forward(self, x):
        y = torch.relu(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = torch.relu(self.conv2(nn.functional.pad(y, (self.pad, 0))))
        return torch.relu(y + self.down(x))


class TCN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        layers, cin = [], n_features
        for cout, d in zip(CHANNELS, DILATIONS):
            layers.append(CausalBlock(cin, cout, KERNEL, d))
            cin = cout
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(CHANNELS[-1], n_classes)

    def forward(self, x):  # x shape: [B, features, WINDOW]
        h = self.blocks(x)
        return self.fc(h[:, :, -1])  # Extract last time step


def sign_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_features(path):
    df = pd.read_csv(path, usecols=lambda c: c.strip() in FEATURES, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Error: Missing required columns in CSV: {missing}")
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan  # Convert Inf to NaN
    return x


def fit_stats(raw):
    miss = np.isnan(raw).mean(axis=0)
    print("[*] Missing fraction per feature:", dict(zip(FEATURES, np.round(miss, 6))))
    if (miss > MAX_MISSING_FRACTION).any():
        raise SystemExit("Error: A feature exceeds the 5% missing threshold.")
    median = np.nanmedian(raw, axis=0)
    z = transform(raw, median, None, None)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return median, mean, std


def transform(raw, median, mean, std):
    filled = np.where(np.isnan(raw), median, raw)  # Impute missing values with training median
    z = sign_log(filled)
    if mean is None:
        return z
    return (z - mean) / std


def c_array(name, arr):
    flat = np.asarray(arr, dtype=np.float64).ravel()
    body = ", ".join(f"{v:.9g}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{{body}}};\n"


def main():
    parser = argparse.ArgumentParser(description="Export TCN model weights to a C++ header file.")
    parser.add_argument("--train-csv", required=True, help="Path to the training dataset CSV.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained PyTorch model checkpoint (.pt).")
    parser.add_argument("--classes", required=True, help="Comma-separated class names.")
    parser.add_argument("--out", default="model_weights.h", help="Output C++ header file path.")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    n_classes = len(classes)

    print("[*] Loading training features and computing preprocessing statistics...")
    raw = load_features(args.train_csv)
    median, mean, std = fit_stats(raw)
    z = transform(raw, median, mean, std)

    print("[*] Loading PyTorch TCN model architecture and weights...")
    model = TCN(len(FEATURES), n_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, len(FEATURES), WINDOW))  # Initialize weight-norm parameters if applicable

    n_params = sum(p.numel() for p in model.parameters())
    expected = 6544 + 33 * n_classes
    print(f"[*] Parameter count: {n_params} (Expected: {expected} for {n_classes} classes)")
    if n_params != expected:
        print("[!] Warning: Model architecture deviates from the standard paper configuration.")

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    lines = [
        "// model_weights.h - Automatically generated by export_weights.py. Do not modify manually.",
        "#pragma once",
        "",
        f"static const int N_FEAT = {len(FEATURES)};",
        f"static const int WIN_LEN = {WINDOW};",
        f"static const int N_CLASS = {n_classes};",
        f"static const int C1 = {CHANNELS[0]};",
        f"static const int C2 = {CHANNELS[1]};",
        f"static const int KSIZE = {KERNEL};",
        f"static const int DIL1 = {DILATIONS[0]};",
        f"static const int DIL2 = {DILATIONS[1]};",
        "static const char* const FEATURE_NAMES[N_FEAT] = {"
        + ", ".join(f'"{f}"' for f in FEATURES) + "};",
        "static const char* const CLASS_NAMES[N_CLASS] = {"
        + ", ".join(f'"{c}"' for c in classes) + "};",
        "",
        c_array("SCALER_MEDIAN", median),
        c_array("SCALER_MEAN", mean),
        c_array("SCALER_STD", std),
        "",
    ]

    blocks = {
        "B1_C1": "blocks.0.conv1", "B1_C2": "blocks.0.conv2", "B1_DS": "blocks.0.down",
        "B2_C1": "blocks.1.conv1", "B2_C2": "blocks.1.conv2", "B2_DS": "blocks.1.down",
    }
    for name, prefix in blocks.items():
        lines.append(c_array(f"{name}_W", sd[prefix + ".weight"]))
        lines.append(c_array(f"{name}_B", sd[prefix + ".bias"]))
    lines.append(c_array("FC_W", sd["fc.weight"]))
    lines.append(c_array("FC_B", sd["fc.bias"]))
    lines.append("")

    # Verification vector: first 16 preprocessed training rows in channel-major format [8][16]
    win = z[:WINDOW]
    x = torch.tensor(win.T[None], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x).numpy()[0]
    lines.append(c_array("CHECK_INPUT", win.T))
    lines.append(c_array("CHECK_LOGITS", logits))

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"[✔] Successfully generated header file: {args.out}")


if __name__ == "__main__":
    main()
