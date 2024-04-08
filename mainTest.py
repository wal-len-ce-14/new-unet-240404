# # totest
from test_model import test_model
for i in range(1, 5):
    test_model("./model/segUnet_loss0.18acc95.77%.pth", f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png")
    test_model("./model/segUnet+_loss0.17acc93.16%.pth", f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png")
    test_model("./model/segUnet++_loss0.19acc93.26%.pth", f"./totest/images/benign ({i}).png", f"./totest/masks/benign ({i})_mask.png")
    
    test_model("./model/segUnet_loss0.18acc95.77%.pth", f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png")
    test_model("./model/segUnet+_loss0.17acc93.16%.pth", f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png")
    test_model("./model/segUnet++_loss0.19acc93.26%.pth", f"./totest/images/malignant ({i}).png", f"./totest/masks/malignant ({i})_mask.png")