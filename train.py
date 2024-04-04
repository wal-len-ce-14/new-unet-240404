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
    max = 94
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loss = []
    test_loss = []
    acc = []
    start = time.time()
    end = time.time()
    # dataset
    try:
        dataset = data(img, mask)
        train_data, test_data = random_split(dataset, [int(len(dataset)*0.85), len(dataset) - int(len(dataset)*0.85)], generator=torch.Generator().manual_seed(42))
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

    # train loop
    for e in range(1, epochs+1):
        epoch_loss = 0
        t_loss = 0
        
    # try:
        print(f"loop {e}:")
        model = model.to(device)
        from tqdm import tqdm
        for idx, (x, y) in tqdm(enumerate(train_Loader), total=len(train_Loader)):
            # forword
            x = x.to(device=device,dtype=torch.float32)
            y = y.to(device=device,dtype=torch.float32)
            p = torch.sigmoid(model(x))
            # backward
            optimizer.zero_grad()
            if(p.shape[1] > 1):
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
            else:
                loss = loss_f(p, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        train_loss += [(epoch_loss/len(train_Loader))]
        print(f"\t[+] epoch_loss = {epoch_loss/len(train_Loader)}")

        for idx, (x, y) in enumerate(test_Loader):
            x = x.to(device=device,dtype=torch.float32)
            y = y.to(device=device,dtype=torch.float32)
            p = torch.sigmoid(model(x))
            if(p.shape[1] > 1):
                # loss1 = loss_f(p[:,0,:,:].unsqueeze(1), y)
                # loss2 = loss_f(p[:,1,:,:].unsqueeze(1), y)
                # loss3 = loss_f(p[:,2,:,:].unsqueeze(1), y)
                # loss4 = loss_f(p[:,3,:,:].unsqueeze(1), y)
                t_loss += loss_f(p[:,3,:,:].unsqueeze(1), y).cpu().item()
            else:
                # loss = loss_f(p, y)
                t_loss += loss_f(p,y).cpu().item()

            p = torch.where(p > 0.5, 1., 0.)
            tr = torch.where(p == y, 1, 0)
            if(idx == 0):
                acc += [((tr.sum() / tr.numel()) *100).to("cpu")]
        if(tr.sum()/tr.numel()*100 > max and (t_loss/len(test_Loader)) < 0.15):
            torch.save(model, f'./model/seg{name}_loss{round(float(t_loss/len(test_Loader)),2)}acc{round(float((tr.sum() / tr.numel()) *100), 2)}%.pth') 
            max = round(float((tr.sum() / tr.numel()) *100), 2)
            print("\t[+] save")
        # print(idx, len(test_Loader))
        test_loss += [(t_loss/len(test_Loader))]
        print(f"\t[+] test loss = {t_loss/len(test_Loader)}")
        print(f"\t[+] acc = {round(float((tr.sum() / tr.numel()) *100), 2)}")
    # except Exception as e:
    #     print(f"4 error in {e} loop:")
    #     print(e)
    #     break
    end = time.time()

    print(f"Complete training. take {(end-start) // 60}")

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    acc = np.array(acc)

    # plt.plot(train_loss ,label="train_loss", color='red')
    # plt.plot(test_loss, label="test_loss", color='blue')
    # plt.grid(True)
    # plt.legend()
    # plt.xticks(range(1, len(train_loss)+1, 5))
    # plt.title(name)
    # plt.show()

    # plt.plot(acc ,label="acc", color='red')
    # plt.grid(True)
    # plt.legend()
    # plt.xticks(range(1, len(acc)+1, 5))
    # plt.title(name)
    # plt.show()

    return train_loss, test_loss, acc


if __name__ == "__main__":
    model = Unet(1,1)
    train(model, 32, 1e-5, "./img/images/", "./img/masks")