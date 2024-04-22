import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import cv2 as cv
import matplotlib.pyplot as plt

from dataset import data
from Net import Unet
from tool import countdice, countiou


import os
import time

def train(
        model,
        batch,
        lr,
        img,
        mask,
        epochs=100,
        name=''
):
    
    #init var
    max = 77
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loss = []
    test_loss = []
    acc = []
    dice = []
    iou = []
    start = time.time()
    end = time.time()
    # dataset
    try:
        dataset = data(img, mask)
        train_data, test_data = random_split(dataset, [int(len(dataset)*0.85), len(dataset) - int(len(dataset)*0.85)])
        train_Loader = DataLoader(train_data, batch, shuffle=True, drop_last=True)
        test_Loader = DataLoader(test_data, batch, shuffle=False, drop_last=True)
    except Exception as e:
        print(f"1 error in")
        print(e)
        

    # set func
    try:
        optimizer = optim.Adam(model.parameters(), lr)
        loss_f = nn.BCELoss()
    except Exception as e:
        print(f"2 error in")
        print(e)

    print(f"\n****Start training {name} model****\n")
    # train loop
    for e in range(1, epochs+1):
        epoch_loss = 0
        t_loss = 0
        
    # try:
        # print(f"loop {e}:")
        model = model.to(device)
        # train
        from tqdm import tqdm
        for idx, (x, y) in tqdm(enumerate(train_Loader), total=len(train_Loader)):
            # forword
            x = x.to(device=device,dtype=torch.float32)
            y = y.to(device=device,dtype=torch.float32)
            p = torch.sigmoid(model(x))
            # backward
            optimizer.zero_grad()
            if(p.shape[1] == 4):
                optimizer.zero_grad()
                loss1 = loss_f(p[:,0,:,:].unsqueeze(1), y)
                loss2 = loss_f(p[:,1,:,:].unsqueeze(1), y)
                loss3 = loss_f(p[:,2,:,:].unsqueeze(1), y)
                loss4 = loss_f(p[:,3,:,:].unsqueeze(1), y)
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward(retain_graph=True)
                loss4.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss4.item()
            elif(p.shape[1] == 2):
                optimizer.zero_grad()
                loss1 = loss_f(p[:,0,:,:].unsqueeze(1), y)
                loss2 = loss_f(p[:,1,:,:].unsqueeze(1), y)
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss2.item()
            else:
                loss = loss_f(p, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        train_loss += [(epoch_loss/len(train_Loader))]      # save loss
        print(f"\t[+] epoch {e} loss: {epoch_loss/len(train_Loader)}")
        # test
        acc_inepoch = 0
        dice_inepoch = 0
        iou_inepoch = 0
        for idx, (x, y) in enumerate(test_Loader):
            x = x.to(device=device,dtype=torch.float32)
            y = y.to(device=device,dtype=torch.float32)
            p = torch.sigmoid(model(x))
            if(p.shape[1] == 4):
                p = p[:,3,:,:].unsqueeze(1)
                t_loss += loss_f(p, y).cpu().item()
            elif(p.shape[1] == 2):
                p = p[:,1,:,:].unsqueeze(1)
                t_loss += loss_f(p, y).cpu().item()
            else:
                t_loss += loss_f(p,y).cpu().item()

            p = torch.where(p > 0.5, 1., 0.)
            tr = torch.where(p == y, 1, 0)

            acc_inepoch += (tr.sum() / tr.numel())
            dice_inepoch += countdice(p, y)
            iou_inepoch += countiou(p, y)

        acc += [acc_inepoch.to("cpu")*100 / len(test_Loader)]
        dice += [dice_inepoch.to("cpu")*100 / len(test_Loader)]
        iou += [iou_inepoch.to("cpu")*100 / len(test_Loader)]
        print(f"\t[+] test loss: {t_loss/len(test_Loader)}")
        print(f"\t[+] dice: {dice[-1]}")
        print(f"\t[+] iou: {iou[-1]}")
        # save model
        if(float(dice[-1]) > 70 and (t_loss/len(test_Loader)) < 0.2 and e > 20 and iou[-1] > 50):
            torch.save(model, f'./model/seg{name}_dice{round(float(dice[-1]), 2)}.pth') 
            print("\t[+] save")
        test_loss += [(t_loss/len(test_Loader))]        # save loss
       

    end = time.time()

    print(f"Complete training. take {(end-start) // 60}")
    
    # save loss
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    acc = np.array(acc)                                         
    dice = np.array(dice)
    iou = np.array(iou)         


    return train_loss, test_loss, acc, dice, iou


if __name__ == "__main__":
    model = Unet(1,1)
    train(model, 32, 1e-3, "./img/images/", "./img/masks")