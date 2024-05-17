import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tool import countdice, countiou
from Net import resNet, CNN
import os

img_H = 224
img_W = 224
device = "cpu"

def test_res_model2(model_path, testingimgdir, file_name):
    imgdir = os.listdir(testingimgdir)
    image_batch = torch.empty(10,1,224,224)
    image_O = np.zeros((10,224,224,1))
    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    plt.setp(axes, xticks=[], yticks=[])
    model = torch.load(model_path).to(device=device)

    for i, im in enumerate(imgdir):
        image = os.path.join(testingimgdir, im)
        image = torch.tensor(np.array(cv.resize(cv.imread(image, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0)
        # print(image.shape)
        image_O[i] = image.cpu().permute(1,2,0).detach().numpy()
        # print(image_O.shape)
        image_batch[i] = image

    show1 = torch.sigmoid(model(image_batch.to(device=device, dtype=torch.float32)))
    # print(model(image.to(device=device, dtype=torch.float32)))
    show = torch.where(show1 > 0.5, 1, 0).detach().numpy()
    # print(show)
    for i, m in enumerate(imgdir):
        # print(image_O.shape)
        axes[0].imshow(image_O[i], cmap="gray")
        axes[0].set_title(f"Original Image\n{m.split('.')[0]}", fontsize=10)
        axes[1].imshow(image_O[i], cmap="gray")
        if(np.array_equal(show[i], [1,0])):
            axes[1].set_title(f"Predict\n{show1[i][0]*100:.2f}%-benign", fontsize=10)
        elif(np.array_equal(show[i], [0,1])):
            axes[1].set_title(f"Predict\n{show1[i][1]*100:.2f}%-malignant", fontsize=10)
        # plt.show()
        plt.savefig("plt-test/" + file_name + f'_{i}' + ".png")




def test_res_model(model_path, testing_img, file_name, f=0):
    image = torch.tensor(np.array(cv.resize(cv.imread(testing_img, cv.IMREAD_GRAYSCALE), (img_H, img_W)))).unsqueeze(0).unsqueeze(0)
    image_O = image.cpu().detach().numpy().squeeze()
    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    plt.setp(axes, xticks=[], yticks=[])
    model = torch.load(model_path).to(device=device)
    # model = resNet(1,2)
    # model.load_state_dict(torch.load(model_path))
    print(image.shape)
    show = torch.sigmoid(model(image.to(device=device, dtype=torch.float32)))
    print(model(image.to(device=device, dtype=torch.float32)))
    show = torch.where(show > 0.5, 1, 0).detach().numpy()
    axes[0].imshow(image_O, cmap="gray")
    axes[0].set_title(f"Original Image\n{testing_img.split('/')[-1].split('.')[0]}", fontsize=10)
    axes[1].imshow(image_O, cmap="gray")
    if(np.array_equal(show, [[1,0]])):
        axes[1].set_title("Predict\nbenign", fontsize=10)
    elif(np.array_equal(show, [[0,1]])):
        axes[1].set_title("Predict\nmalignant", fontsize=10)
    # plt.show()
    plt.savefig("plt-test/" + file_name + ".png")


def test_model(model_path, testing_img, testing_mask, file_name, f=0):

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
        if not f:
            model = torch.load(m).to(device=device)
        else:
            model = model_path
        show = torch.sigmoid(model(image.to(device=device, dtype=torch.float32)))
        if show.shape[1] > 1:
            show = show[:,-1,:,:]
        # show = torch.where(show > 0.85, 255, 0)
        # l1 = torch.where(show > 0.75, 90, 0)
        show = torch.where(show > 0.8, 255, 0)
        # l3 = torch.where(show > 0.5, 60, 0)
        # l4 = torch.where(show > 0.4, 40, 0)
        # show = l1 + l2 + l3 + l4



        image_P = show.cpu().detach().numpy().squeeze()
        name = testing_img.split('/')[-1].split('.')[0] + ' --' + m.split('/')[-1].split('_')[0]
        plt.suptitle(name)
        axes[idx+2].imshow(image_P, cmap="gray")  
        
        dice=countdice(torch.where(show>1, 1, 0), torch.where(mask>1, 1, 0))
        iou=countiou(torch.where(show>1, 1, 0), torch.where(mask>1, 1, 0))
        dice = "{:.2f}".format(dice)
        iou = "{:.2f}".format(iou)

        axes[idx+2].set_title(f"Predict use {m.split('/')[-1].split('_')[0].replace('seg','')}\ndiec={dice}\niou={iou}", fontsize=10)
        
    # plt.show()
    plt.savefig("plt-test/" + file_name + ".png")

if __name__ == "__main__":  
    test_model(["./model/segUnet_loss0.18acc95.77%.pth"], "./totest/images/benign (2).png", "./totest/masks/benign (2)_mask.png", "test-1")