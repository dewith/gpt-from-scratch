"""
This module provides functions for evaluating the model. The `estimate_loss`
function is used to estimate the loss of the model on the training and
validation sets.
"""

import torch


@torch.no_grad()
def estimate_loss(model, batcher, eval_iters):
    """Estimate the loss of the model on the training and validation sets."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = batcher.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
