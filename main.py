from Net import Unet, UnetPlus, UnetPlusPlus
from train import train
import matplotlib.pyplot as plt

# modelU = Unet(1,1)
# modelUP = UnetPlus(1,1)
# modelUPP = UnetPlusPlus(1,1)

# UPP_trainl, UPP_loss, UPP_acc = train(modelUPP, 16, 1e-4, "./img/images/", "./img/masks", epochs=30, name="Unet++")
# UP_trainl, UP_loss, UP_acc = train(modelUP, 16, 1e-4, "./img/images/", "./img/masks", epochs=30, name="Unet+")
# U_trainl, U_loss, U_acc = train(modelU, 16, 1e-4, "./img/images/", "./img/masks", epochs=30, name="Unet")

# # show results

# plt.plot(U_trainl ,label="U-train_loss", color='red')
# plt.plot(U_loss, label="U-test_loss", color='blue')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(U_trainl), 5))
# plt.title("Unet-loss")
# plt.show()

# plt.plot(UP_trainl ,label="UP-train_loss", color='red')
# plt.plot(UP_loss, label="UP-test_loss", color='blue')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(UP_trainl), 5))
# plt.title("UnetP-loss")
# plt.show()

# plt.plot(UPP_trainl ,label="UPP-train_loss", color='red')
# plt.plot(UPP_loss, label="UPP-test_loss", color='blue')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(UPP_trainl), 5))
# plt.title("UnetPP-loss")
# plt.show()

# # show virsus

# plt.plot(UPP_loss ,label="UPP-loss", color='red')
# plt.plot(UP_loss, label="UP-loss", color='green')
# plt.plot(U_loss, label="U-loss", color='blue')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(UPP_trainl), 5))
# plt.title("Unet vs Unet+ vs Unet++ - loss")
# plt.show()

# plt.plot(UPP_acc ,label="UPP-acc", color='red')
# plt.plot(UP_acc, label="UP-loss", color='green')
# plt.plot(U_acc, label="U-acc", color='blue')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(UPP_trainl), 5))
# plt.title("Unet vs Unet+ vs Unet++ - acc")
# plt.show()

from test_model import test_model
for i in range(1, 5):
    test_model("./model/segUnet_loss0.18acc95.77%.pth", f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png")
    test_model("./model/segUnet+_loss0.17acc93.16%.pth", f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png")
    test_model("./model/segUnet++_loss0.19acc93.26%.pth", f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png")
    
    test_model("./model/segUnet_loss0.18acc95.77%.pth", f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png")
    test_model("./model/segUnet+_loss0.17acc93.16%.pth", f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png")
    test_model("./model/segUnet++_loss0.19acc93.26%.pth", f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png")