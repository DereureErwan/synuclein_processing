# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F


def deepclusterv2_loss_func(
    outputs: torch.Tensor, assignments: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes DeepClusterV2's loss given a tensor containing logits from multiple views
    and a tensor containing cluster assignments from the same multiple views.

    Args:
        outputs (torch.Tensor): tensor of size PxVxNxC where P is the number of prototype
            layers and V is the number of views.
        assignments (torch.Tensor): tensor of size PxVxNxC containing the assignments
            generated using k-means.
        temperature (float, optional): softmax temperature for the loss. Defaults to 0.1.

    Returns:
        torch.Tensor: DeepClusterV2 loss.
    """
    loss = 0
    for h in range(outputs.size(0)):
        scores = outputs[h].view(-1, outputs.size(-1)) / temperature
        targets = assignments[h].repeat(outputs.size(1)).to(outputs.device, non_blocking=True)
        loss += F.cross_entropy(scores, targets, ignore_index=-1)
    return loss / outputs.size(0)

def supervised_contrastive_loss(features, labels, temperature=0.1):
    device = features.device

    # Mask out samples with label == -1
    valid_mask = labels != -1
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    if len(labels) < 2:
        # Not enough valid samples to compute loss
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    batch_size = features.shape[0]

    # Normalize features
    features = F.normalize(features, dim=1)

    # Cosine similarity matrix
    similarity_matrix = torch.matmul(features, features.T)  # [N, N]
    logits = similarity_matrix / temperature

    # Create mask for positive pairs (same label, not self)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask = mask.fill_diagonal_(0)

    # For numerical stability
    logits_max = torch.max(logits, dim=1, keepdim=True)[0]
    logits = logits - logits_max.detach()

    # Compute log prob (excluding self)
    exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # Average over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    loss = -mean_log_prob_pos.mean()
    return loss