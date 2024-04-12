# # totest
from test_model import test_model
import matplotlib.pyplot as plt

testID = 4


model = [
    "model/segUnet_3_loss0.17dice6987.23%.pth",
    "model/segUnet+_3_loss0.15dice6854.19%.pth",
    "model/segUnet++_3_loss0.15dice7816.39%.pth",
    "model/segMyUnet_3_loss0.17dice7644.79%.pth"
]

for i in range(1, 6):
    test_model(model, f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png", f"test{testID}_{(i-1)*2+1}")     
    test_model(model, f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png", f"test{testID}_{(i-1)*2+2}")

    # plt.show()
