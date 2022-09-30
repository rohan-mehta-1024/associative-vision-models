import torch
import argparse
import losses as l
import torch.nn as nn
from data import LoadData 
from simclr import SimCLR
import torch.nn.functional as F
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLRTrainDataTransform as DefaultTrain,
    SimCLREvalDataTransform as DefaultEval, 
)

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("data_aug")
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("save_as", type=int)
args = parser.parse_args()

class FineTune(nn.Module):
    """Projection module for SimCLR (Pytorch Lightning implementation)"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.linear = nn.Linear(2048, 5)
        
    def forward(self, x):
        return self.linear(self.model(x))

def train(model, data_aug, batch_size, epochs, save_as):
    
    if model == "simclr"
        if data_aug == "default":
            data = LoadData([DefaultTrain(100), DefaultEval(100)]).generate_split_dataloader()
        else:
            data = LoadData([LoadData.random_masking_transform()]).generate_split_dataloader()
    else:
        data = LoadData([LoadData.default_transform()]).generate_dataloader()
    masked_test_data = LoadData([LoadData.default_transform()], "LFW_masked").generate_dataloader()
        
    if model == "simclr":
        model = torch.jit.script(SimCLR(batch_size, len(data(1, "train")), epochs=epochs))
        optimizer, scheduler = model.configure_optimizer()
        loss_fn = l. NT_Xent(batch_size)
        mode = "SSL"
    elif "simclr" in model:
        model = torch.jit.script(SimCLR(32, len(loader(1, "train")), epochs=epochs))
        model.load_state_dict(torch.load(model))
        model = FineTune(model)
        mode = " "
    else: pass

    train = data(batch_size, "train")
    val = data(batch_size, "val")
    test = data(batch_size, "test")
    masked_test = data(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        loss_l, acc_l = [], []
              
        for (data, labels) in train:
            optimizer.zero_grad()
            if mode == "SSL":
                logits = [model(i.to(device)) for i in data]
                loss = loss_fn(*logits)
            else:
                logits = model(data.to(device))
                loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            loss_l.append(loss.item())
            
            if mode != "SSL":
                acc_l.append(get_acc(logits, l))
                
        wandb.log({"train_loss" : torch.mean(torch.tensor(loss_l)), "epoch" : epoch})
        if mode != "SSL":
            wandb.log({"train_acc" : torch.mean(torch.tensor(acc_l)), "epoch" : epoch})
            
        vloss_l, vacc_l = [], []
        for (data, labels) in val:
            if mode == "SSL":
                logits = [model(i.to(device)) for i in data]
                loss = loss_fn(*logits)
            else:
                logits = model(data.to(device))
                loss = loss_fn(logits, labels)
                
            vloss_l.append(loss.item())
            if mode != "SSL":
                vacc_l.append(get_acc(logits, l))
                
        wandb.log({"val_loss" : torch.mean(torch.tensor(vloss_l)), "epoch" : epoch})
        if mode != "SSL":
            wandb.log({"val_acc" : torch.mean(torch.tensor(vacc_l)), "epoch" : epoch})
        scheduler.step()

        if mode != "SSL":
            tloss_l, tacc_l = [], []
            for (data, labels) in test:
                logits = model(data.to(device))
                loss = loss_fn(logits, labels)

                tloss_l.append(loss.item())
                tacc_l.append(get_acc(logits, l))
                
            wandb.log({"unmasked_test_loss" : torch.mean(torch.tensor(tloss_l))})
            wandb.log({"unmasked_test_loss" : torch.mean(torch.tensor(tacc_l))})
            
            tloss_l, tacc_l = [], []
            for (data, labels) in masked_test:
                logits = model(data.to(device))
                loss = loss_fn(logits, labels)

                tloss_l.append(loss.item())
                tacc_l.append(get_acc(logits, l))
                
            wandb.log({"unmasked_test_loss" : torch.mean(torch.tensor(tloss_l))})
            wandb.log({"unmasked_test_loss" : torch.mean(torch.tensor(tacc_l))})
    torch.save(model.state_dict(), f"{save_as}.pt")

if __name__ == "__main__":
    train(model, data_aug, batch_size, epochs, save_as)