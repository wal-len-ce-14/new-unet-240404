import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_H = 224
img_W = 224
device = "cuda" if torch.cuda.is_available() else "cpu"


def test_model(model_path, testing_img, testing_mask, file_name):

    image = torch.tensor(np.array(cv.resize(cv.imread(testing_img, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(np.array(cv.resize(cv.imread(testing_mask, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0).unsqueeze(0)
    image_O = image.cpu().detach().numpy().squeeze()
    image_M = mask.cpu().detach().numpy().squeeze()

    fig, axes = plt.subplots(1, model_path.__len__()+2, figsize=((model_path.__len__()+2)*3, 5))
    plt.setp(axes, xticks=[], yticks=[])
    axes[0].imshow(image_O, cmap="gray")
    axes[0].set_title("Original Image", fontsize=10)
    axes[1].imshow(image_M, cmap="gray")
    axes[1].set_title("Ground truth Mask", fontsize=10)

    for idx, m in enumerate(model_path):
        model = torch.load(m).to(device=device)
        show = torch.sigmoid(model(image.to(device=device, dtype=torch.float32)))
        if show.shape[1] > 1:
            show = show[:,-1,:,:]
        show = torch.where(show > 0.5, 255, 0)
        image_P = show.cpu().detach().numpy().squeeze()
        name = testing_img.split('/')[-1].split('.')[0] + ' --' + m.split('/')[-1].split('_')[0]
        plt.suptitle(name)
        axes[idx+2].imshow(image_P, cmap="gray")  
        axes[idx+2].set_title(f"Predict use {m.split('/')[-1].split('_')[0].replace('seg','')}", fontsize=10)
    # plt.show()
    plt.savefig("plt-test/" + file_name + ".png")
    # return plt
    # cv.imshow("image",  cv.imread(testing_img, cv.IMREAD_GRAYSCALE))

if __name__ == "__main__":  
    test_model(["./model/segUnet_loss0.18acc95.77%.pth"], "./totest/images/benign (2).png", "./totest/masks/benign (2)_mask.png", "test-1")