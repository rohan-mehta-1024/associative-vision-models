import torch
import torch.nn as nn
from torchvision.models import resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class Projection(nn.Module):
    """Projection module for SimCLR (Pytorch Lightning implementation)"""
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

class SimCLR(torch.nn.Module):
    def __init__(self,
                 batch_size, 
                 num_samples, 
                 lr=1e-4, 
                 wt_decay=1e-6, 
                 warmup_epochs=10, 
                 epochs=100):
        super().__init__()
        self.lr = lr
        self.wt_decay = wt_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        
        self.encoder = resnet50()
        self.encoder.fc = torch.nn.Identity()
        self.projection = Projection()
        
        self.train_iters_per_epoch = num_samples // batch_size

    @staticmethod
    @torch.jit.ignore
    def exclude_from_wt_decay(named_params, skip_list=['bias', 'bn']):
        """Pytorch Lightning implementation"""
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params},
            {'params': excluded_params, 'weight_decay': 0.}
        ]
    
    @torch.jit.ignore
    def configure_optimizer(self):
        new_params = SimCLR.exclude_from_wt_decay(self.named_parameters())
        optimizer = LARS(new_params, lr=self.lr, weight_decay=self.wt_decay)
        
        warmup_epochs = self.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.epochs * self.train_iters_per_epoch
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )
        
        return optimizer, scheduler
        
    def forward(self, x):
        return self.encoder(x)

    def training_step(self, xi, xj):
        xi = self.projection(self.encoder(xi))
        xj = self.projection(self.encoder(xj))
        return xi, xj

    def val_step(self, xi, xj):
        xi = self.projection(self.encoder(xi))
        xj = self.projection(self.encoder(xj))
        return xi, xj