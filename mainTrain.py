from Net import Unet, UnetPlus, UnetPlusPlus, MyUnet
from train import train
import matplotlib.pyplot as plt

testID = 3

modelU = Unet(1,1)
modelUP = UnetPlus(1,1)
modelUPP = UnetPlusPlus(1,1)
modelMy = MyUnet(1,1)

UPP_trainl, UPP_loss, UPP_acc, UPP_dice, UPP_iou = train(modelUPP, 10, 1e-3, "./img/images/", "./img/masks", epochs=30, name=f"Unet++_{testID}")
UP_trainl, UP_loss, UP_acc, UP_dice, UP_iou = train(modelUP, 10, 1e-3, "./img/images/", "./img/masks", epochs=30, name=f"Unet+_{testID}")
U_trainl, U_loss, U_acc, U_dice, U_iou = train(modelU, 10, 1e-3, "./img/images/", "./img/masks", epochs=30, name=f"Unet_{testID}")
M_trainl, M_loss, M_acc, M_dice, M_iou = train(modelMy, 10, 1e-3, "./img/images/", "./img/masks", epochs=30, name=f"MyUnet_{testID}")

# show results

p1 = plt.plot(U_trainl ,label="U-train_loss", color='red')
p1.plot(U_loss, label="U-test_loss", color='blue')
p1.grid(True)
p1.legend()
p1.xticks(range(0, len(U_trainl), 5))
p1.title("Unet-loss")
p1.savefig(f"plt/Unet-loss_{testID}.png")
# p1.show()

p2 = plt.plot(UP_trainl ,label="UP-train_loss", color='red')
p2.plot(UP_loss, label="UP-test_loss", color='blue')
p2.grid(True)
p2.legend()
p2.xticks(range(0, len(UP_trainl), 5))
p2.title("UnetP-loss")
p2.savefig(f"plt/UnetP-loss_{testID}.png")
# plt.show()

p3 = plt.plot(UPP_trainl ,label="UPP-train_loss", color='red')
p3.plot(UPP_loss, label="UPP-test_loss", color='blue')
p3.grid(True)
p3.legend()
p3.xticks(range(0, len(UPP_trainl), 5))
p3.title("UnetPP-loss")
p3.savefig(f"plt/UnetPP-loss_{testID}.png")
# plt.show()

p6 = plt.plot(M_trainl ,label="M-train_loss", color='red')
p6.plot(M_loss, label="M-test_loss", color='blue')
p6.grid(True)
p6.legend()
p6.xticks(range(0, len(M_trainl), 5))
p6.title("MyUnet-loss")
p6.savefig(f"plt/MyUnet-loss_{testID}.png")
# plt.show()

# show virsus

p4 = plt.plot(UPP_loss ,label="UPP-loss", color='red')
p4.plot(UP_loss, label="UP-loss", color='green')
p4.plot(U_loss, label="U-loss", color='blue')
p4.plot(M_loss, label="M-loss", color='black')
p4.grid(True)
p4.legend()
p4.xticks(range(0, len(UPP_trainl), 5))
p4.title("Unet vs Unet+ vs Unet++ vs M-loss --loss")
p4.savefig(f"plt/Unet-vs-UnetP-vs-UnetPP-vs-M-loss_{testID}.png")
# plt.show()

p5 = plt.plot(UPP_acc ,label="UPP-acc", color='red')
p5.plot(UP_acc, label="UP-loss", color='green')
p5.plot(U_acc, label="U-acc", color='blue')
p5.plot(M_acc, label="M-acc", color='black')
p5.grid(True)
p5.legend()
p5.xticks(range(0, len(UPP_trainl), 5))
p5.title("Unet vs Unet+ vs Unet++ vs M-acc --acc")
p5.savefig(f"plt/Unet-vs-UnetP-vs-UnetPP-vs-M-acc_{testID}.png")
# plt.show()

p7 = plt.plot(UPP_dice ,label="UPP-dice", color='red')
p7.plot(UP_dice, label="UP-dice", color='green')
p7.plot(U_dice, label="U-dice", color='blue')
p7.plot(M_dice, label="M-dice", color='black')
p7.grid(True)
p7.legend()
p7.xticks(range(0, len(UPP_trainl), 5))
p7.title("Unet vs Unet+ vs Unet++ vs M-dice --dice")
p7.savefig(f"plt/Unet-vs-UnetP-vs-UnetPP-vs-M-dice_{testID}.png")
# plt.show()

p8 = plt.plot(UPP_iou ,label="UPP-iou", color='red')
p8.plot(UP_iou, label="UP-iou", color='green')
p8.plot(U_iou, label="U-iou", color='blue')
p8.plot(M_iou, label="M-iou", color='black')
p8.grid(True)
p8.legend()
p8.xticks(range(0, len(UPP_trainl), 5))
p8.title("Unet vs Unet+ vs Unet++ vs M-iou --iou")
p8.savefig(f"plt/Unet-vs-UnetP-vs-UnetPP-vs-M-iou_{testID}.png")
# plt.show()