import os
import torch.utils.data
import torch
from src.utils import *
from src.grid_divider import *
import numpy as np

from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.transforms import build_transforms

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_folder: str, annot_path: str):
        
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)
        self.classes = {'credential': 0, 'noncredential': 1}
        
    def __getitem__(self, item: int):
        image_file = list(set(self.paths))[item] # image path
        img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file] # box coordinates
        img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file] # box types
        
        if len(img_coords) == 0:
            raise IndexError('list index out of range')
            
        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential
        grid_arr = read_img(img_path=os.path.join(self.img_folder, image_file+'.png'),
                            coords=img_coords, types=img_classes, grid_num=10)
        
        return {"input": grid_arr, 'target':img_label}
    
    def __len__(self):
        return self.num_imgs
    


@register_dataset("web_dataset")
class MyClassyDataset(ClassyDataset):
    
    def __init__(self, img_folder, annot_path, batchsize_per_replica, shuffle, num_samples):
        
        dataset = MyDataset(img_folder=img_folder,
                            annot_path=annot_path)
        super().__init__(dataset=dataset, batchsize_per_replica=batchsize_per_replica, 
                         shuffle=shuffle, transform=None, num_samples=num_samples)
        
    @classmethod
    def from_config(cls, config):
        return cls(
            img_folder = config["img_folder"],
            annot_path = config["annot_path"],
            batchsize_per_replica=config["batchsize_per_replica"],
            shuffle=config["shuffle"],
            num_samples=config["num_samples"]
        )
