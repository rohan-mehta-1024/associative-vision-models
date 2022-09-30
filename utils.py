import torch
import torch.nn.functional as F

def get_acc(logits, labels):
    argmax = torch.argmax(logits, dim=1)
    return torch.mean((labels == argmax).float())