import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, num_classes=5):
        super().__init__()
        if alpha is None: alpha = torch.ones(num_classes)
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        at = self.alpha[targets]
        return (at * (1-pt)**self.gamma * ce).mean()
