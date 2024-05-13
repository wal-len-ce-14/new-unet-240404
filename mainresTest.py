from test_model import test_model, test_res_model2, test_res_model
import matplotlib.pyplot as plt

testID = 14

model = 'model/segresnet_14_dice77.19.pth'

# for i in range(1, 6):
test_res_model2(model, f"./totest/images", f"test{testID}")     
# test_res_model(model, f"./totest/images/malignant ({1}).png", f"test{testID}_2")