# # totest
from test_model import test_model
import matplotlib.pyplot as plt
col = 6
row = 40

model = [
    "model/segUnet_2_loss0.15acc94.44%.pth",
    "model/segUnet+_2_loss0.13acc93.58%.pth",
    "model/segUnet++_2_loss0.14acc92.82%.pth"
]

# fig, axs = plt.subplots(10, model.__len__()+2)
# plt.setp(axs, xticks=[], yticks=[])
for i in range(1, 6):
    test_model(model, f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png", f"test{(i-1)*2+1}-0")     
    test_model(model, f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png", f"test{(i-1)*2+2}-0")

    # plt.show()
