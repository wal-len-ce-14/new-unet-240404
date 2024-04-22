from Net import Unet, UnetPlus, UnetPlusPlus, MyUnet
from train import train
import matplotlib.pyplot as plt

testID = 8

modelU = Unet(1,1)
modelUP = UnetPlus(1,1)
modelUPP = UnetPlusPlus(1,1)
modelMy = MyUnet(1,1)

UPP_trainl, UPP_loss, UPP_acc, UPP_dice, UPP_iou = train(modelUPP, 16, 1e-4, "./img/images/", "./img/masks", epochs=30, name=f"Unet++_{testID}")
UP_trainl, UP_loss, UP_acc, UP_dice, UP_iou = train(modelUP, 16, 1e-4, "./img/images/", "./img/masks", epochs=30, name=f"Unet+_{testID}")
U_trainl, U_loss, U_acc, U_dice, U_iou = train(modelU, 16, 1e-4, "./img/images/", "./img/masks", epochs=30, name=f"Unet_{testID}")
M_trainl, M_loss, M_acc, M_dice, M_iou = train(modelMy, 16, 1e-4, "./img/images/", "./img/masks", epochs=30, name=f"MyUnet_{testID}")

# show results

plt.plot(U_trainl ,label="U-train_loss", color='red')
plt.plot(U_loss, label="U-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(U_trainl), 5))
plt.title("Unet-loss")
plt.savefig(f"plt/_{testID}/Unet-loss_{testID}.png")
plt.close()
# p1.show()

plt.plot(UP_trainl ,label="UP-train_loss", color='red')
plt.plot(UP_loss, label="UP-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UP_trainl), 5))
plt.title("UnetP-loss")
plt.savefig(f"plt/_{testID}/UnetP-loss_{testID}.png")
plt.close()
# plt.show()

plt.plot(UPP_trainl ,label="UPP-train_loss", color='red')
plt.plot(UPP_loss, label="UPP-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("UnetPP-loss")
plt.savefig(f"plt/_{testID}/UnetPP-loss_{testID}.png")
plt.close()
# plt.show()

plt.plot(M_trainl ,label="M-train_loss", color='red')
plt.plot(M_loss, label="M-test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(M_trainl), 5))
plt.title("MyUnet-loss")
plt.savefig(f"plt/_{testID}/MyUnet-loss_{testID}.png")
plt.close()
# plt.show()

# show virsus

plt.plot(UPP_loss ,label="UPP-loss", color='red')
plt.plot(UP_loss, label="UP-loss", color='green')
plt.plot(U_loss, label="U-loss", color='blue')
plt.plot(M_loss, label="M-loss", color='black')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("Unet V.S. Unet+ V.S. Unet++ V.S. M --loss")
plt.savefig(f"plt/_{testID}/Unet-vs-UnetP-vs-UnetPP-vs-M-loss_{testID}.png")
plt.close()
# plt.show()

plt.plot(UPP_acc ,label="UPP-acc", color='red')
plt.plot(UP_acc, label="UP-acc", color='green')
plt.plot(U_acc, label="U-acc", color='blue')
plt.plot(M_acc, label="M-acc", color='black')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("Unet V.S. Unet+ V.S. Unet++ V.S. M --acc")
plt.savefig(f"plt/_{testID}/Unet-vs-UnetP-vs-UnetPP-vs-M-acc_{testID}.png")
plt.close()
# plt.show()

plt.plot(UPP_dice ,label="UPP-dice", color='red')
plt.plot(UP_dice, label="UP-dice", color='green')
plt.plot(U_dice, label="U-dice", color='blue')
plt.plot(M_dice, label="M-dice", color='black')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("Unet V.S. Unet+ V.S. Unet++ V.S. M --dice")
plt.savefig(f"plt/_{testID}/Unet-vs-UnetP-vs-UnetPP-vs-M-dice_{testID}.png")
plt.close()
# plt.show()

plt.plot(UPP_iou ,label="UPP-iou", color='red')
plt.plot(UP_iou, label="UP-iou", color='green')
plt.plot(U_iou, label="U-iou", color='blue')
plt.plot(M_iou, label="M-iou", color='black')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(UPP_trainl), 5))
plt.title("Unet V.S. Unet+ V.S. Unet++ V.S. M --iou")
plt.savefig(f"plt/_{testID}/Unet-vs-UnetP-vs-UnetPP-vs-M-iou_{testID}.png")
plt.close()
# plt.show()

plt.plot(U_iou, label="U-iou", color='blue')
plt.plot(M_iou, label="M-iou", color='black')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(U_iou), 5))
plt.title("Unet V.S. M --iou")
plt.savefig(f"plt/_{testID}/Unet-vs-M-iou_{testID}.png")
plt.close()
# plt.show()

plt.plot(U_dice, label="U-dice", color='blue')
plt.plot(M_dice, label="M-dice", color='black')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(U_iou), 5))
plt.title("Unet V.S. M --dice")
plt.savefig(f"plt/_{testID}/Unet-vs-M-dice_{testID}.png")
plt.close()
# plt.show()