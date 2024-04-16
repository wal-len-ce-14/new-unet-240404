# # totest
from test_model import test_model
import matplotlib.pyplot as plt

testID = 6


model = [
    "model/segUnet_6_dice74.57.pth",
    "model/segUnet+_6_dice76.9.pth",
    "model/segUnet++_6_dice76.51.pth",
    "model/segMyUnet_6_dice74.59.pth"
]

for i in range(1, 6):
    test_model(model, f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png", f"test{testID}_{(i-1)*2+1}")     
    test_model(model, f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png", f"test{testID}_{(i-1)*2+2}")

