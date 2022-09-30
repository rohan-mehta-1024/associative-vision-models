import torch
import torch.nn as nn

def angular_softmax_loss():
    """Loss function for SphereFace"""
    pass

def large_margin_cos_loss():
    """Loss function for CosFace"""
    pass

def additive_angular_margin_loss():
    """Loss function for ArcFace"""
    
def triplet_loss():
    """Loss function for FaceNet"""
    
class NT_Xent(nn.Module):
    """Loss function used for SimCLR (Mix of PyTorch Lightning and Spijkervet implementations)"""
    def __init__(self, batch_size, temperature=0.5):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.N = batch_size * 2
        
        self.mask = torch.eye(self.N).bool()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.cos_sim = nn.CosineSimilarity(dim=2)

    def __call__(self, x_i, x_j):
        z = torch.cat((x_i, x_j), dim=0)
        sim = self.cos_sim(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        #extract positive pairs: (x_i, x_j)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.N, 1)
        
        # remove left digonal (x_i, x_i) to get negative pairs
        negative_samples = sim[self.mask].reshape(self.N, -1)
        labels = torch.zeros(self.N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / self.N
        return loss