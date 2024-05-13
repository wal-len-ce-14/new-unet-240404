from Net import resNet, CNN
from train_resnet import train
import matplotlib.pyplot as plt

testID = 14

res = resNet(1,2)
cnn = CNN(1,2)

R_trainl, R_loss, R_acc, R_dice, R_iou, R_recall = train(res, 10, 1e-5, "./img/images/", epochs=30, name=f"resnet_{testID}")
# C_trainl, C_loss, C_acc, C_dice, C_iou, C_recall = train(cnn, 32, 2e-5, "./img/images/", epochs=30, name=f"cnn_{testID}")


# show results
plt.clf()
plt.plot(R_trainl ,label="train_loss", color='red')
plt.plot(R_loss, label="test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("resnet-loss")
plt.savefig(f"plt/_{testID}/resnet-loss_{testID}.png")
plt.close()

plt.clf()
plt.plot(R_dice ,label="test_dice", color='red')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("resnet-dice")
plt.savefig(f"plt/_{testID}/resnet-dice_{testID}.png")
plt.close()

plt.clf()
plt.plot(R_iou ,label="resnet_iou", color='red')  
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("resnet-iou")
plt.savefig(f"plt/_{testID}/resnet-iou_{testID}.png")
plt.close()

plt.clf()
plt.plot(R_acc ,label="resnet_acc", color='red')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("resnet-acc")
plt.savefig(f"plt/_{testID}/resnet-acc_{testID}.png")
plt.close()

plt.clf()
plt.plot(R_recall ,label="resnet_recall", color='red')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("resnet-recall")
plt.savefig(f"plt/_{testID}/resnet-recall_{testID}.png")
plt.close()

# from test_model import test_model, test_res_model

# for i in range(1, 6):
#     test_res_model(res, f"./totest/images/benign ({i}).png", f"test{testID}_{(i-1)*2+1}")     
#     test_res_model(res, f"./totest/images/malignant ({i}).png", f"test{testID}_{(i-1)*2+2}")
# show cnn
# plt.clf()
# plt.plot(C_trainl ,label="train_loss", color='red')
# plt.plot(C_loss, label="test_loss", color='blue')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(C_trainl), 5))
# plt.title("CNN-loss")
# plt.savefig(f"plt/_{testID}/CNN-loss_{testID}.png")
# plt.close()

# plt.clf()
# plt.plot(C_dice ,label="CNN_dice", color='red')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(C_trainl), 5))
# plt.title("CNN-dice")
# plt.savefig(f"plt/_{testID}/CNN-dice_{testID}.png")
# plt.close()

# plt.clf()
# plt.plot(C_iou ,label="CNN_iou", color='red')  
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(C_trainl), 5))
# plt.title("CNN-iou")
# plt.savefig(f"plt/_{testID}/CNN-iou_{testID}.png")
# plt.close()

# plt.clf()
# plt.plot(C_acc ,label="CNN_acc", color='red')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(C_trainl), 5))
# plt.title("CNN-acc")
# plt.savefig(f"plt/_{testID}/CNNt-acc_{testID}.png")
# plt.close()

# plt.clf()
# plt.plot(C_recall ,label="CNN_recall", color='red')
# plt.grid(True)
# plt.legend()
# plt.xticks(range(0, len(C_trainl), 5))
# plt.title("CNN-recall")
# plt.savefig(f"plt/_{testID}/CNN-recall_{testID}.png")
# plt.close()