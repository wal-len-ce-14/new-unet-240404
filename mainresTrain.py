from Net import resNet
from train_resnet import train
import matplotlib.pyplot as plt

testID = 8

res = resNet(1,2)

R_trainl, R_loss, R_acc, R_dice, R_iou = train(res, 32, 1e-3, "./img/images/", epochs=100, name=f"resnet_{testID}")

# show results

plt.plot(R_trainl ,label="train_loss", color='red')
plt.plot(R_loss, label="test_loss", color='blue')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("resnet-loss")
plt.savefig(f"plt/_{testID}/resnet-loss_{testID}.png")
plt.close()

plt.plot(R_dice ,label="test_dice", color='red')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("test-dice")
plt.savefig(f"plt/_{testID}/test-dice_{testID}.png")
plt.close()

plt.plot(R_iou ,label="test_iou", color='red')  
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("test-iou")
plt.savefig(f"plt/_{testID}/test-iou_{testID}.png")
plt.close()

plt.plot(R_acc ,label="test_acc", color='red')
plt.grid(True)
plt.legend()
plt.xticks(range(0, len(R_trainl), 5))
plt.title("test-acc")
plt.savefig(f"plt/_{testID}/test-acc_{testID}.png")
plt.close()