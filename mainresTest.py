from test_model import test_model, test_res_model
import matplotlib.pyplot as plt

testID = 8

model = 'model/segresnet_9_dice92.19.pth'

for i in range(1, 6):
    test_res_model(model, f"./totest/images/benign ({i}).png", f"test{testID}_{(i-1)*2+1}")     
    test_res_model(model, f"./totest/images/malignant ({i}).png", f"test{testID}_{(i-1)*2+2}")