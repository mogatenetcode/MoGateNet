import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, loss_fns, weights=None, deep_weights=None):
        super().__init__()

        self.loss_fns = nn.ModuleList(loss_fns)

        if weights is None:
            self.weights = [1.0 for _ in loss_fns]
        else:
            self.weights = weights

        if deep_weights is None:
            self.deep_weights = [1.0, 0.4, 0.4, 0.4]
        else:
            self.deep_weights = deep_weights

    def forward(self, outputs, targets):
        if isinstance(outputs, (list, tuple)):
            output_list = outputs
        else:
            output_list = [outputs]

        total_loss = 0.0

        for i, output in enumerate(output_list):
            if output.shape[2:] != targets.shape[2:]:
                output = F.interpolate(
                    output,
                    size=targets.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )

            output_loss = 0.0
            for loss_fn, weight in zip(self.loss_fns, self.weights):
                output_loss = output_loss + weight * loss_fn(output, targets)

            deep_weight = self.deep_weights[min(i, len(self.deep_weights) - 1)]
            total_loss = total_loss + deep_weight * output_loss

        return total_loss
