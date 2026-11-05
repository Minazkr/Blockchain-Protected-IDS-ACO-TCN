# train.py
# -------------------------------------------------
# Training the TCN model
# Special attention is given to rare attack classes
# -------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Give higher importance to rare attacks.
    
    Attacks that appear less frequently in the data 
    receive larger weights. This forces the model to 
    pay more attention to minority classes instead of 
    ignoring them.
    """
    class_ids, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    n_classes = len(class_ids)

    # Basic inverse frequency weighting
    weights = total / (n_classes * counts)

    # Prevent extremely large weights
    weights = np.clip(weights, 0.5, 10.0)

    # Create a weight tensor in the correct class order
    weight_list = [0.0] * n_classes
    for cls, w in zip(class_ids, weights):
        weight_list[int(cls)] = w

    return torch.tensor(weight_list, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    labels_for_weights: np.ndarray,
    epochs: int = 30,
    lr: float = 0.001,
    device: str = "cpu"
) -> Dict:
    """
    Train the model using weighted loss for rare attacks.
    """
    model.to(device)

    # Calculate weights for rare classes
    class_weights = calculate_class_weights(labels_for_weights)
    class_weights = class_weights.to(device)

    # Loss function with class weights
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_score = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Here we would evaluate on validation set
        # and keep the best model

    return {
        "best_val_score": best_val_score,
        "model_state": best_weights
    }
