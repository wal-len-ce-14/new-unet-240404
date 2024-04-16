import os
import cv2 as cv 
import torch
from torch.utils.data import Dataset
import numpy as np

img_H = 224
img_W = 224

class data(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = torch.tensor(np.array(cv.resize(cv.imread(img_path, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0)
        mask = torch.tensor(np.array(cv.resize(cv.imread(mask_path, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0)
        mask = torch.where(mask > 127, 1., 0.)
        # print(image.shape)
        return image, mask

class resdata(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        if "benign" in self.images[index]:
            mask = torch.tensor([1,0])
        elif "malignant" in self.images[index]:
            mask = torch.tensor([0,1])
        else:
            print("data error")
        image = torch.tensor(np.array(cv.resize(cv.imread(img_path, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0)
        
        return image, mask

if __name__ == "__main__":
    img = resdata("./img/images")
    i, m = img.__getitem__(1)
    print(i.shape, m.shape)