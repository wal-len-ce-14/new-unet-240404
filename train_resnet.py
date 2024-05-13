import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time

from dataset import resdata
from Net import resNet
from tool import countdice, countiou

def train(
        model,
        batch,
        lr,
        img,
        epochs=100,
        name=''
):
    #init var
    max = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    train_loss = []
    test_loss = []
    acc = []
    dice = []
    iou = []
    recall = []
    start = time.time()
    end = time.time()
    # dataset
    dataset = resdata(img)
    train_data, test_data = random_split(dataset, [int(len(dataset)*0.85), len(dataset) - int(len(dataset)*0.85)])
    train_Loader = DataLoader(train_data, batch, shuffle=True, drop_last=True)
    test_Loader = DataLoader(test_data, batch, shuffle=False, drop_last=True)
    # set func
    optimizer = optim.Adam(model.parameters(), lr)
    loss_f = nn.BCELoss()

    print(f"\n****Start training {name} model****\n")

    # train loop
    
    for e in range(1, epochs+1):
        epoch_loss = 0
        t_loss = 0
        # train
        model.train()
        from tqdm import tqdm
        for idx, (x, y) in tqdm(enumerate(train_Loader), total=len(train_Loader)):
            # forword
            x = x.to(device=device,dtype=torch.float32)
            y = y.to(device=device,dtype=torch.float32)
            # print(y.shape)
            p = torch.sigmoid(model(x))
            # backward
            optimizer.zero_grad()
            loss = loss_f(p, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss += [(epoch_loss/len(train_Loader))]      # save loss
        print(f"\t[+] epoch {e} loss: {epoch_loss/len(train_Loader)}")
        # test
        model.train()
        acc_inepoch = 0
        dice_inepoch = 0
        iou_inepoch = 0
        for idx, (x, y) in enumerate(test_Loader):
            x = x.to(device=device,dtype=torch.float32)
            y = y.to(device=device,dtype=torch.float32)
            p = torch.sigmoid(model(x))
            loss = loss_f(p, y)
            p = torch.where(p > 0.5, 1., 0.)
            tr = torch.where(p == y, 1, 0)
            # print("p",p[-10:],"\ny", y[-10:])
            t_loss += loss.item()
            acc_inepoch += (tr.sum() / tr.numel()).to('cpu')
            dice_inepoch += countdice(p, y)
            iou_inepoch += countiou(p, y)
        from sklearn.metrics import recall_score, confusion_matrix
        if e > 10000:
            import seaborn as sns
            import matplotlib.pyplot as plt
            y_b = torch.where(torch.all(torch.eq(y, torch.tensor([[1,0]]).to(device)), dim=1), 1, 0)
            y_m = torch.where(torch.all(torch.eq(y, torch.tensor([[0,1]]).to(device)), dim=1), -1, 0)
            yy = y_b+y_m
            p_b = torch.where(torch.all(torch.eq(p, torch.tensor([[1,0]]).to(device)), dim=1), 1, 0)
            p_m = torch.where(torch.all(torch.eq(p, torch.tensor([[0,1]]).to(device)), dim=1), -1, 0)
            pp = p_b+p_m

            cm = confusion_matrix(yy.to('cpu'), pp.to('cpu'))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(f'plt/Confusion Matrix_{e}_{name}')
        print("p => ", p[-8:])
        print("y => ", y[-8:])
        recall += [recall_score(y.to('cpu'), p.to('cpu'), average=None)[1]*100]
        acc += [acc_inepoch*100 / len(test_Loader)]
        dice += [dice_inepoch.to("cpu")*100 / len(test_Loader)]
        iou += [iou_inepoch.to("cpu")*100 / len(test_Loader)]
        print(f"\t[+] test loss: {t_loss/len(test_Loader)}")
        print(f"\t[+] dice: {dice[-1]}")
        print(f"\t[+] iou: {iou[-1]}")
        print(f"\t[+] recall: {recall[-1]}")
        # save model
        if dice[-1]*iou[-1]*recall[-1]/(t_loss/(len(test_Loader)+0.00001)) > max:
            # torch.save(model, f'./model/seg{name}_dice{round(float(dice[-1]), 2)}.pth') 
            torch.save(model, f'./model/seg{name}_dice{round(float(dice[-1]), 2)}.pth')
            max = dice[-1]*iou[-1]*recall[-1]/(t_loss/len(test_Loader))
            print("\t[+] save")
        test_loss += [(t_loss/len(test_Loader))]   
    end = time.time()
    print(f"Complete training. take {(end-start) // 60}")
    recall = np.array(recall)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    acc = np.array(acc)                                         
    dice = np.array(dice)
    iou = np.array(iou)  

    return train_loss, test_loss, acc, dice, iou, recall

if __name__ == "__main__":
    model = resNet(1,2)
    train(model, 32, 1e-3, "./img/images/", epochs=3)

