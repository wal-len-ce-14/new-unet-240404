from Net import Unet, UnetPlus, UnetPlusPlus
from train import train
import matplotlib.pyplot as plt

modelU = Unet(1,1)
modelUP = UnetPlus(1,1)
modelUPP = UnetPlusPlus(1,1)

UPP_trainl, UPP_loss, UPP_acc = train(modelUPP, 16, 1e-3, "./img/images/", "./img/masks", epochs=30, name="Unet++_2")
UP_trainl, UP_loss, UP_acc = train(modelUP, 16, 1e-3, "./img/images/", "./img/masks", epochs=30, name="Unet+_2")
U_trainl, U_loss, U_acc = train(modelU, 16, 1e-3, "./img/images/", "./img/masks", epochs=30, name="Unet_2")

# show results

plt.plot(U_trainl ,label="U-train_loss", color='red')
plt.plot(U_loss, label="U-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(U_trainl), 5))
plt.title("Unet-loss")
plt.show()

plt.plot(UP_trainl ,label="UP-train_loss", color='red')
plt.plot(UP_loss, label="UP-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UP_trainl), 5))
plt.title("UnetP-loss")
plt.show()

plt.plot(UPP_trainl ,label="UPP-train_loss", color='red')
plt.plot(UPP_loss, label="UPP-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("UnetPP-loss")
plt.show()

# show virsus

plt.plot(UPP_loss ,label="UPP-loss", color='red')
plt.plot(UP_loss, label="UP-loss", color='green')
plt.plot(U_loss, label="U-loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("Unet vs Unet+ vs Unet++ - loss")
plt.show()

plt.plot(UPP_acc ,label="UPP-acc", color='red')
plt.plot(UP_acc, label="UP-loss", color='green')
plt.plot(U_acc, label="U-acc", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("Unet vs Unet+ vs Unet++ - acc")
plt.show()