# model.py
# -------------------------------------------------
# Lightweight Causal Temporal Convolutional Network (TCN)
# -------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Convolution that only looks at current and past time steps.
    This keeps the model causal (no future information leakage).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )

    def forward(self, x):
        # Pad only on the left side (past)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    One residual block of the TCN.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.dropout(F.relu(self.conv1(x)))
        out = self.dropout(F.relu(self.conv2(out)))

        return F.relu(out + residual)


class TCN(nn.Module):
    """
    Lightweight Temporal Convolutional Network.
    
    Input shape:  [batch, channels, sequence_length]
    Output:       class logits
    """

    def __init__(
        self,
        num_inputs: int,
        num_classes: int,
        channels=(16, 32),
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(channels)

        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation = 2 ** i

            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        y = self.network(x)

        # Take the last time step
        y = y[:, :, -1]

        return self.classifier(y)
