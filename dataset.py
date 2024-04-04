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

if __name__ == "__main__":
    img = data("./img/images", "./img/masks")
    img.__getitem__(1)