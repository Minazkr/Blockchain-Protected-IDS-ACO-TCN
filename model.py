# model.py
# -------------------------------------------------
# Lightweight Causal TCN for Intrusion Detection
# -------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    A convolution that only uses current and past information.
    It never looks at future time steps (causal).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

    def forward(self, x):
        # Add padding only on the left (past side)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    One building block of the TCN.
    Contains two causal convolutions + residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

        # Used when input and output channels are different
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.residual_conv is None else self.residual_conv(x)

        out = F.relu(self.conv1(x))
        out = self.dropout(out)

        out = F.relu(self.conv2(out))
        out = self.dropout(out)

        return F.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network.
    
    Input:  [batch_size, num_features, sequence_length]
    Output: class predictions
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        channels=(16, 32),
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        layers = []
        for i, out_channels in enumerate(channels):
            in_channels = num_features if i == 0 else channels[i - 1]
            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # Extract temporal features
        features = self.feature_extractor(x)

        # Use only the last time step for classification
        last_step = features[:, :, -1]

        return self.classifier(last_step)
