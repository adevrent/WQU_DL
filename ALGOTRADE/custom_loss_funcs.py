import torch

def sMAPE_loss(forecast, target, eps=1e-8):
    # sMAPE = 200/H * mean(|F - y| / (|F| + |y|))
    numerator = torch.abs(forecast - target)
    denominator = torch.abs(forecast) + torch.abs(target) + eps
    return 200 * torch.mean(numerator / denominator)