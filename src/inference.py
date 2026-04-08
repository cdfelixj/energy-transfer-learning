"""
Monte Carlo Dropout Inference Utilities

Provides uncertainty-aware prediction for trained EnergyLSTM models by running
multiple stochastic forward passes with Dropout kept active at inference time.

Usage
-----
    from src.inference import predict_with_uncertainty

    mean, std, lower, upper = predict_with_uncertainty(model, x_tensor)
"""

import torch
import numpy as np


def predict_with_uncertainty(
    model,
    x: torch.Tensor,
    n_samples: int = 50,
    confidence: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Monte Carlo Dropout inference to produce uncertainty estimates.

    Sets the model to eval mode (disables BatchNorm running stats, etc.) but
    re-enables only Dropout modules so each forward pass is stochastic.

    Args:
        model:      A trained EnergyLSTM (or compatible) model.
        x:          Input tensor of shape (batch, seq_len, features).
        n_samples:  Number of stochastic forward passes.
        confidence: Width of the confidence interval, e.g. 0.9 → 90% CI.

    Returns:
        mean      — shape (batch,), mean prediction across samples.
        std       — shape (batch,), standard deviation across samples.
        lower_ci  — shape (batch,), lower bound of the confidence interval.
        upper_ci  — shape (batch,), upper bound of the confidence interval.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # keep Dropout stochastic; everything else stays in eval mode

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)          # (batch, 1)
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds, axis=1) if preds[0].ndim > 1 else np.stack(preds, axis=0)
    # Ensure shape is (n_samples, batch)
    preds = np.array([p.squeeze(-1) if p.ndim == 2 else p for p in preds])  # (n_samples, batch)

    alpha = (1.0 - confidence) / 2.0
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    lower_ci = np.quantile(preds, alpha, axis=0)
    upper_ci = np.quantile(preds, 1.0 - alpha, axis=0)

    return mean, std, lower_ci, upper_ci


def predict_batch_with_uncertainty(
    model,
    dataloader,
    n_samples: int = 50,
    confidence: float = 0.9,
    device: str = "cpu",
) -> dict:
    """Run MC Dropout inference over an entire DataLoader.

    Accumulates predictions and uncertainty estimates across all batches.

    Args:
        model:      A trained EnergyLSTM (or compatible) model.
        dataloader: PyTorch DataLoader yielding (x, y) batches.
        n_samples:  Number of stochastic forward passes per batch.
        confidence: Width of the confidence interval.
        device:     Device to run inference on ('cpu' or 'cuda').

    Returns:
        dict with keys:
            'mean'       — np.ndarray (N,), mean predictions.
            'std'        — np.ndarray (N,), standard deviation.
            'lower_ci'   — np.ndarray (N,), lower CI bound.
            'upper_ci'   — np.ndarray (N,), upper CI bound.
            'actuals'    — np.ndarray (N,), ground truth values.
    """
    model = model.to(device)
    all_mean, all_std, all_lower, all_upper, all_actual = [], [], [], [], []

    for x, y in dataloader:
        x = x.to(device)
        mean, std, lower, upper = predict_with_uncertainty(model, x, n_samples, confidence)
        all_mean.append(mean)
        all_std.append(std)
        all_lower.append(lower)
        all_upper.append(upper)
        all_actual.append(y.squeeze(-1).cpu().numpy())

    return {
        'mean': np.concatenate(all_mean),
        'std': np.concatenate(all_std),
        'lower_ci': np.concatenate(all_lower),
        'upper_ci': np.concatenate(all_upper),
        'actuals': np.concatenate(all_actual),
    }
