from Net import Unet, UnetPlus, UnetPlusPlus
from train import train
import threading

# modelU = Unet(1,1)
# modelUP = UnetPlus(1,1)
modelUPP = UnetPlusPlus(1,1)

# train(modelU, 10, 1e-3, "./img/images/", "./img/masks", epochs=10, name="Unet")
# train(modelUP, 10, 1e-3, "./img/images/", "./img/masks", epochs=10, name="Unet+")
train(modelUPP, 10, 1e-3, "./img/images/", "./img/masks", epochs=10, name="Unet++")