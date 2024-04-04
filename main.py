from Net import Unet, UnetPlus, UnetPlusPlus
from train import train
import matplotlib.pyplot as plt

modelU = Unet(1,1)
# modelUP = UnetPlus(1,1)
modelUPP = UnetPlusPlus(1,1)

# train(modelU, 10, 1e-3, "./img/images/", "./img/masks", epochs=10, name="Unet")
# train(modelUP, 10, 1e-3, "./img/images/", "./img/masks", epochs=10, name="Unet+")
UPP_trainl, UPP_loss, UPP_acc = train(modelUPP, 16, 1e-3, "./img/images/", "./img/masks", epochs=30, name="Unet++")
U_trainl, U_loss, U_acc = train(modelU, 64, 1e-3, "./img/images/", "./img/masks", epochs=30, name="Unet")

plt.plot(UPP_trainl ,label="UPP-train_loss", color='red')
plt.plot(UPP_loss, label="UPP-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(1, len(UPP_trainl)+1, 5))
plt.title("UnetPP-loss")
plt.show()

plt.plot(U_trainl ,label="U-train_loss", color='red')
plt.plot(U_loss, label="U-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(1, len(UPP_trainl)+1, 5))
plt.title("Unet-loss")
plt.show()

plt.plot(UPP_loss ,label="UPP-loss", color='red')
plt.plot(U_loss, label="U-loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(1, len(UPP_trainl)+1, 5))
plt.title("Unet vs Unet++ - loss")
plt.show()

plt.plot(UPP_acc ,label="UPP-acc", color='red')
plt.plot(U_acc, label="U-acc", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(1, len(UPP_trainl)+1, 5))
plt.title("Unet vs Unet++ - acc")
plt.show()