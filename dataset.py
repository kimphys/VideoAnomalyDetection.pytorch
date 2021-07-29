import torch
from torch.utils.data import Dataset

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

import glob

class SequenceDataset(Dataset):
    def __init__(self, channels, size, frames_dir, time_steps, test=False):
        self.test = test

        self.time_steps = time_steps
        self.frames_list = glob.glob(frames_dir + '/**/*.tif', recursive=True)
        self.frames_list.sort()
        self.size = size

        self.channels = channels


    def __getitem__(self, index):

        if self.channels == 1:
            imgs = [cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) for frame in self.frames_list[index:index + self.time_steps]]
        else: 
            imgs = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32) for frame in self.frames_list[index:index + self.time_steps]]
        
        o_seqs = [self.simple_transform(img, self.size) for img in imgs]
        seqs = [self.base_transform(img, self.size) for img in imgs]

        seqs = torch.stack(seqs)

        return seqs, o_seqs

    def base_transform(self, o_img, size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):

        transform = A.Compose([
                                A.Resize(height=size, 
                                        width=size, 
                                        always_apply=True, 
                                        p=1.0),
                                A.Normalize(mean=mean,
                                            std=std,
                                            max_pixel_value=255.0,
                                            p=1.0),
                                ToTensorV2(p=1.0)
                            ], p=1.0)

        img = transform(image=o_img)['image']

        return img

    def simple_transform(self, o_img, size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    
        transform = A.Compose([
                                A.Resize(height=size, 
                                        width=size, 
                                        always_apply=True, 
                                        p=1.0),
                                ToTensorV2(p=1.0)
                            ], p=1.0)

        img = transform(image=o_img)['image']

        return img

    def __len__(self):                    
        return len(self.frames_list) - self.time_steps