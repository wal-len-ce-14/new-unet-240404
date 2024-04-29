from Net import resNet
from train_resnet import train
import matplotlib.pyplot as plt

testID = 10

res = resNet(1,2)

R_trainl, R_loss, R_acc, R_dice, R_iou, R_recall = train(res, 32, 1e-4, "./img/images/", epochs=25, name=f"resnet_{testID}")

# print(R_trainl, R_loss, R_acc, R_dice, R_iou, R_recall)

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
plt.title("test-dice")
plt.savefig(f"plt/_{testID}/test-dice_{testID}.png")
plt.close()

plt.clf()
plt.plot(R_iou ,label="test_iou", color='red')  
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("test-iou")
plt.savefig(f"plt/_{testID}/test-iou_{testID}.png")
plt.close()

plt.clf()
plt.plot(R_acc ,label="test_acc", color='red')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("test-acc")
plt.savefig(f"plt/_{testID}/test-acc_{testID}.png")
plt.close()

plt.clf()
plt.plot(R_recall ,label="test_recall", color='red')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("test-recall")
plt.savefig(f"plt/_{testID}/test-recall_{testID}.png")
plt.close()