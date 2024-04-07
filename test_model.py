import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_H = 224
img_W = 224
device = "cuda" if torch.cuda.is_available() else "cpu"


def test_model(model_path, testing_img, testing_mask):
    model = torch.load(model_path).to(device=device)
    image = torch.tensor(np.array(cv.resize(cv.imread(testing_img, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(np.array(cv.resize(cv.imread(testing_mask, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0).unsqueeze(0)
    
    show = torch.sigmoid(model(image.to(device=device, dtype=torch.float32)))
    if show.shape[1] > 1:
        show = show[:,3,:,:]
    
    # ss = torch.where(show > 0.5, 120, 0)
    # ss = torch.where(show > 0.7, 180, 0)
    show = torch.where(show > 0.95, 180, 0)
    name = testing_img.split('/')[-1].split('.')[0] + ' --' + model_path.split('/')[-1].split('_')[0]
    fig, axes = plt.subplots(1, 3)
    plt.suptitle(name)
    plt.title("testing_img")
    image1 = show.cpu().detach().numpy().squeeze()
    image2 = mask.cpu().detach().numpy().squeeze()
    image3 = image.cpu().detach().numpy().squeeze()
    axes[0].imshow(image3, cmap="gray")
    axes[0].set_title("Original Image")
    axes[1].imshow(image1, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[2].imshow(image2, cmap="gray")
    axes[2].set_title("ground truth Mask")
    plt.show()
    # cv.imshow("image",  cv.imread(testing_img, cv.IMREAD_GRAYSCALE))

if __name__ == "__main__":  
    test_model("./model/segUnet_loss0.18acc95.77%.pth", "./totest/images/benign (2).png", "./totest/masks/benign (2)_mask.png")