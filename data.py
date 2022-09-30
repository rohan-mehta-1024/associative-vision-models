import os
import glob
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
from pydash import flatten_deep as flatten
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform  

class GenericDataset(Dataset):
    
    def __init__(self, data, 
                 labels, 
                 transforms, 
                 train_split, 
                 val_split, 
                 test_split):
        
        self.data = data
        self.labels = labels
        self.transforms = transforms 
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.setup()
    
    def setup(self):
        for i in self.train_split:
            self.data[i] = self.transforms[0](self.data[i])
        
        for i in self.val_split:
            if len(self.transforms) >= 2:
                t = self.transforms[1]
            else: t = self.transforms[-1]
            self.data[i] = t(self.data[i])
            
        for i in self.test_split:
            if len(self.transforms) >= 3:
                t = self.transforms[2]
            else: t = self.transforms[-1]
            self.data[i] = t(self.data[i])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LoadData():
    def __init__(self, transforms, img_dir="LFW", splits=[0.8, 0.1]):
        super().__init__()
        np.random.seed(42)
        
        self.splits = splits
        self.img_dir = img_dir
        
        self.transforms = [T.Compose([Image.open, t]) for t in transforms]
        self.label_mapping = {i[1] : i[0] for i in enumerate(os.listdir(img_dir))}
        
    @staticmethod
    def mask_lower_half_of_image(img, value, amount=0.5):
        nth_pixel = img[0].shape[0]
        half = int(nth_pixel * (1 - amount))
        minus_half = nth_pixel - half

        if value == 0:
            random = torch.rand(3,1)
            random=random.unsqueeze(1).repeat(1, half, nth_pixel)
            img[:, minus_half:, :]=random
            return img
        elif value == 1:
            random = torch.rand((3, half, nth_pixel))
            img[:, minus_half:, :]=random
            return img
        elif value == 2:
            ones = torch.ones(3, half, nth_pixel)
            zeros = torch.zeros(3, minus_half, nth_pixel)
            mask = torch.concat((ones, zeros), 1) == 0 
            mask = torch.reshape(mask, (3, nth_pixel, nth_pixel))
            return img.masked_fill(mask, 0)
        
    @staticmethod
    def random_masking_transform():
        mask_fn = lambda x: UnmaskedFaces.mask_lower_half_of_image(x, np.random.choice(2,1))
        pseudo_masking_transform = T.Compose([
            T.Resize(100),
            T.ToTensor(),
            T.Lambda(mask_fn)
        ])
        return lambda x: (x, pseudo_masking_transform(x))
    
    @staticmethod
    def default_transform():
        return T.Compose([
            T.Resize(100),
            T.ToTensor()
        ])
    
    @staticmethod
    def default_simclr_train():
        return T.Compose([
            SimCLRTrainDataTransform(100),
            T.Lambda(lambda x: x[:2])
        ])
    
    @staticmethod
    def default_simclr_eval():
        return T.Compose([
            SimCLREvalDataTransform(100),
            T.Lambda(lambda x: x[:2])
        ])

    def get_name(self, path):
        no_digits = lambda x: not any([i.isdigit() for i in x])
        file = path.split("/")[2].split("_")
        
        new_file = []
        for i in list(zip(file, map(no_digits, file))):
            if i[1]:
                new_file.append(i[0])
            else: break
        return "_".join(new_file)
    
    def read_data(self):
        raw_lfw = [glob.glob(f"{self.img_dir}/{person}/*.jpg") 
                   for person in os.listdir(self.img_dir)]
        return raw_lfw
    
    def preprocess(self, sample_list): 
        images, labels = sample_list, [self.get_name(i) for i in sample_list]
        labels = torch.tensor([self.label_mapping[i] for i in labels])
        return images, F.one_hot(labels, len(os.listdir(self.img_dir)))
    
    
    def generate_data_and_splits(self):
        data = flatten(self.read_data())
        
        indices = np.array(list(range(len(data))))
        indices = np.random.permutation(indices)
        data_splits = [int(len(indices) * i) for i in self.splits]
        
        train_split = indices[:data_splits[0]]
        val_split = indices[data_splits[0]:data_splits[0]+data_splits[1]]
        test_split = indices[data_splits[0]+data_splits[1]:]
        return(data, (train_split, val_split, test_split))
        
        return GenericDataset(
            *self.preprocess(data), 
            self.transforms,
            train_split,
            val_split,
            test_split
        )
    
    def generate_dataloader(self):
        (data, splits) = self.generate_data_and_splits()
        
        dataset = GenericDataset(
            *self.preprocess(data), 
            self.transforms,
            *splits
        )
        
        return lambda b: DataLoader(
            dataset, 
            batch_size=b, 
            shuffle=True,
            drop_last=True,
        )
    
    def generate_split_dataloader(self):
        (data, splits) = self.generate_data_and_splits()
        (train_split, val_split, test_split) = splits 
        
        dataset = GenericDataset(
            *self.preprocess(data), 
            self.transforms,
            *splits
        )
        
        return lambda b, m: DataLoader(
            dataset, 
            batch_size=b, 
            sampler=SubsetRandomSampler(
                train_split if m == "train" else
                val_split if m == "val" else
                test_split), drop_last=True,
        )