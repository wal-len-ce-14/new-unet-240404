import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x, concat=0):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat([x2, x], dim=1)
        x = self.conv(x)
        return x
    
class Unet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c

    
        self.conv1to64 = DoubleConv(in_c, 64)
        self.down64to128 = Down(64, 128)
        self.down128to256 = Down(128, 256)
        self.down256to512 = Down(256, 512)
        self.down512to1024 = Down(512, 1024)
        self.up1024to512 = Up(1024, 512)
        self.up512to256 = Up(512, 256)
        self.up256to128 = Up(256, 128)
        self.up128to64 = Up(128, 64)
        self.out64to1 = nn.Conv2d(64, out_c, 1, 1, bias=False)
    
    def forward(self, x):
        # down, encode
        x1 = self.conv1to64(x)      
        x2 = self.down64to128(x1)
        x3 = self.down128to256(x2)
        x4 = self.down256to512(x3)
        x5 = self.down512to1024(x4)

        # up, decode
        y4 = self.up1024to512(x5, x4)
        y3 = self.up512to256(y4, x3)
        y2 = self.up256to128(y3, x2)
        y1 = self.up128to64(y2, x1)
        y = self.out64to1(y1)

        # return
        return y

class UnetPlus(nn.Module):      # 4 out channel
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1to64 = DoubleConv(in_c, 64)
        self.down64to128 = Down(64, 128)
        self.down128to256 = Down(128, 256)
        self.down256to512 = Down(256, 512)
        self.down512to1024 = Down(512, 1024)
        self.up1024to512 = Up(1024, 512)
        self.up512to256 = Up(512, 256)
        self.up256to128 = Up(256, 128)
        self.up128to64 = Up(128, 64)
        self.out64to1 = nn.Conv2d(64, out_c, 1, 1, bias=False)
    def concatall(self, x1, x2 ,x3, x4):
        x4 = self.out64to1(x4)
        x3 = self.out64to1(x3)
        x2 = self.out64to1(x2)
        y = torch.cat([x4, x3, x2, x1], dim=1)
        return y

    def forward(self, x):
        # down, encode
        x1 = self.conv1to64(x)      
        x2 = self.down64to128(x1)
        x3 = self.down128to256(x2)
        x4 = self.down256to512(x3)
        x5 = self.down512to1024(x4)
        # up, decode
        x31 = self.up1024to512(x5, x4)
        x21 = self.up512to256(x4, x3)
        x11 = self.up256to128(x3, x2)
        x01 = self.up128to64(x2, x1)
        x22 = self.up512to256(x31, x21)
        x12 = self.up256to128(x21, x11)
        x02 = self.up128to64(x11, x01)
        x13 = self.up256to128(x22, x12)
        x03 = self.up128to64(x12, x02)
        x04 = self.up128to64(x13, x03)
        y1 = self.out64to1(x04)
        y = self.concatall(y1, x01, x02, x03)
        return y

class UnetPlusPlus(nn.Module):   # 4 out channel
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1to64 = DoubleConv(in_c, 64)
        self.conv768to64 = DoubleConv(768, 64)
        self.conv768to256 = DoubleConv(768, 256)
        self.conv768to128 = DoubleConv(768, 128)
        self.conv384to128 = DoubleConv(384, 128)
        self.conv384to256 = DoubleConv(384, 256)
        self.conv512to128 = DoubleConv(512, 128)
        self.conv512to256 = DoubleConv(512, 256)
        self.conv192to64 = DoubleConv(192, 64)
        self.conv128to64 = DoubleConv(128, 64)
        self.conv256to64 = DoubleConv(256, 64)
        self.conv256to128 = DoubleConv(256, 128)
        self.conv320to64 = DoubleConv(320, 64)

        self.down64to128 = Down(64, 128)
        self.down128to256 = Down(128, 256)
        self.down256to512 = Down(256, 512)
        self.down512to1024 = Down(512, 1024)
        self.up1024to512 = Up(1024, 512)
        self.T512to256 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.T256to128 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.T128to64 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.out64to1 = nn.Conv2d(64, out_c, 1, 1, bias=False)
    def concat(self, x1, x2):
        y = torch.cat([x2, x1], dim=1)
        return y
    def concatall(self, x1, x2 ,x3, x4):
        x4 = self.out64to1(x4)
        x3 = self.out64to1(x3)
        x2 = self.out64to1(x2)
        y = torch.cat([x4, x3, x2, x1], dim=1)
        return y
    def forward(self, x):
        # down, encode
        x1 = self.conv1to64(x)      
        x2 = self.down64to128(x1)
        x3 = self.down128to256(x2)
        x4 = self.down256to512(x3)
        x5 = self.down512to1024(x4)
        # up, decode
        # 1
        x31 = self.up1024to512(x5, x4)
        x21 = self.conv512to256(self.concat(self.T512to256(x4), x3))
        x11 = self.conv256to128(self.concat(self.T256to128(x3), x2))
        x01 = self.conv128to64(self.concat(self.T128to64(x2), x1))
        # 2
        x22 = self.conv768to256(torch.cat([x3, x21, self.T512to256(x31)], dim=1))
        x12 = self.conv384to128(torch.cat([x2, x11, self.T256to128(x21)], dim=1))
        x02 = self.conv192to64(torch.cat([x1, x01, self.T128to64(x11)], dim=1))
        # 3
        x13 = self.conv512to128(torch.cat([self.T256to128(x22), x2, x11, x12], dim=1))
        x03 = self.conv256to64(torch.cat([self.T128to64(x12), x1, x01, x02], dim=1))
        # 4
        x04 = self.conv320to64(torch.cat([self.T128to64(x13), x1, x01, x02, x03], dim=1))
        y1 = self.out64to1(x04)
        # Lambda
        y = self.concatall(y1, x01, x02 ,x03)
        return y

class MyUnet(nn.Module):   # 4 out channel
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1to64 = DoubleConv(in_c, 64)
        self.conv384to128 = DoubleConv(384, 128)
        self.conv192to64 = DoubleConv(192, 64)
        self.conv128to64 = DoubleConv(128, 64)
        self.conv256to128 = DoubleConv(256, 128)
        
        self.down64to128 = Down(64, 128)
        self.down128to256 = Down(128, 256)
        self.down256to512 = Down(256, 512)
        self.down512to1024 = Down(512, 1024)
        self.up1024to512 = Up(1024, 512)
        self.up512to256 = Up(512, 256)
        self.T256to128 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.T128to64 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.out64to1 = nn.Conv2d(64, out_c, 1, 1, bias=False)
        
    def forward(self, x):
        # down, encode
        x1 = self.conv1to64(x)      
        x2 = self.down64to128(x1)
        x3 = self.down128to256(x2)
        x4 = self.down256to512(x3)
        x5 = self.down512to1024(x4)
        # up, decode
        x31 = self.up1024to512(x5, x4)
        x22 = self.up512to256(x31, x3)
        x11 = self.conv256to128(torch.cat([x2, self.T256to128(x3)], dim=1))
        x13 = self.conv384to128(torch.cat([x2, x11, self.T256to128(x22)], dim=1))
        x02 = self.conv128to64(torch.cat([x1, self.T128to64(x11)], dim=1))
        x04 = self.conv192to64(torch.cat([x1, x02, self.T128to64(x13)], dim=1))
        y = torch.cat([self.out64to1(x02), self.out64to1(x04)], dim=1)
        return y

class resdown(nn.Module):
        def __init__(self, in_c, out_c, w=224):  # 224
            super(resdown, self).__init__()
            self.pool = nn.AvgPool2d(w // 32, stride=1)
            self.linear = nn.Linear(in_c, 1024)
            self.out = nn.Linear(1024, out_c)
            self.flat = nn.Flatten(1,-1)
            self.drop = nn.Dropout(0.15)
            self.reLU = nn.ReLU(inplace=True)
        def forward(self, x):
            x1 = self.pool(x)
            x2 = self.drop(self.reLU(self.linear(self.drop(self.reLU(self.linear(self.flat(x1)))))))
            # print(self.flat(x1).shape)
            y = self.out(x2)
            return y

class CNN(nn.Module):
    def __init__(self, in_c, out_c, w=224): 
        super(CNN, self).__init__() 
        self.x1conv3to64 = nn.Conv2d(in_c,64,1,1,0)
        self.x3conv64to128 = nn.Conv2d(64,128,3,1,1)
        self.x3conv128to256 = nn.Conv2d(128,256,3,1,1)
        self.x3conv256to512 = nn.Conv2d(256,512,3,1,1)
        self.x3conv512to1024 = nn.Conv2d(512,1024,3,1,1)
        self.x3conv1024to2048 = nn.Conv2d(1024,2048,3,1,1)
        self.poolx2 = nn.MaxPool2d(2)
        # self.poolavg = nn.AvgPool2d(w // 8, stride=1)
        self.flat = nn.Flatten(1,-1)
        self.drop = nn.Dropout(0.15)
        self.reLU = nn.ReLU(inplace=True)
        self.linear2048to1024 = nn.Linear(2048, 1024)
        self.linear1024to1024 = nn.Linear(1024, 1024)
        self.linear1024to256 = nn.Linear(1024, 256)
        self.linear256to256 = nn.Linear(256, 256)
        self.linearto256 = nn.Linear(200704, 256)
        self.out = nn.Linear(1024, out_c)
        self.out2 = nn.Linear(256, out_c)
    def forward(self, x):
        x = self.poolx2(self.reLU(self.x1conv3to64(x)))
        x = self.poolx2(self.reLU(self.x3conv64to128(x)))
        x = self.poolx2(self.reLU(self.x3conv128to256(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = self.drop(self.reLU(self.linearto256(x)))
        x = self.drop(self.reLU(self.linear256to256(x)))
        x = self.drop(self.reLU(self.linear256to256(x)))
        y = self.drop(self.reLU(self.out2(x)))

        return y

class resNet(nn.Module): # mynet
    def __init__(self, in_c, out_c):
        super(resNet, self).__init__() 
        self.out_c = out_c
        self.x7conv3to64 = nn.Conv2d(in_c,64,7,1,3,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.x1conv64to64 = DoubleConv(64,64)
        self.x1conv64to128 = Down(64,128)
        self.x1conv128to128 = DoubleConv(128,128)
        self.x1conv128to256 = Down(128,256)
        self.x1conv256to256 = DoubleConv(256,256)
        self.x1conv256to512 = Down(256,512)
        self.x1conv512to512 = DoubleConv(512,512)
        self.x1conv512to1024 = Down(512,1024)
        self.down = resdown(1024, out_c)
    def forward(self, x):
        x1 = self.pool(self.x7conv3to64(x))
        x2 = self.x1conv64to64(x1) + x1
        x3 = self.x1conv64to64(x2) + x2
        x4 = self.x1conv64to64(x3) + x3
        x5 = self.x1conv64to128(x4)
        x6 = self.x1conv128to128(x5) + x5
        x7 = self.x1conv128to128(x6) + x6
        x8 = self.x1conv128to128(x7) + x7
        x9 = self.x1conv128to256(x8)
        x10 = self.x1conv256to256(x9) + x9
        x11 = self.x1conv256to256(x10) + x10
        x12 = self.x1conv256to256(x11) + x11
        x13 = self.x1conv256to256(x12) + x12
        x14 = self.x1conv256to512(x13)
        x15 = self.x1conv512to512(x14) + x14
        x16 = self.x1conv512to1024(x15)
        y = self.down(x16)
        return y

class resNet18(nn.Module):
            
    def __init__(self, in_c, out_c):
        super(resNet18, self).__init__() 
        self.out_c = out_c
        self.x7conv3to64 = nn.Conv2d(in_c,64,7,1,3,bias=False)
        self.x1conv64to128 = nn.Conv2d(64,128,1,1,bias=False)
        self.x1conv128to256 = nn.Conv2d(128,256,1,1,bias=False)
        self.x1conv256to512 = nn.Conv2d(256,512,1,1,bias=False)
        self.x1conv512to1028 = nn.Conv2d(512,1028,1,1,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.x1conv64to64 = DoubleConv(64,64)
        self.x1conv64to128 = Down(64,128)
        self.x1conv128to128 = DoubleConv(128,128)
        self.x1conv128to256 = Down(128,256)
        self.x1conv256to256 = DoubleConv(256,256)
        self.x1conv256to512 = Down(256,512)
        self.x1conv512to512 = DoubleConv(512,512)
        self.x1conv512to1024 = Down(512,1024)
        self.down = resdown(1024, out_c)
    def forward(self, x):
        x1 = self.pool(self.x7conv3to64(x))
        # print("0 ", self.pool(x1).shape) 
        x2 = self.x1conv64to64(x1) + x1
        x3 = self.x1conv64to64(x2) + x2
        # print("1 ", self.pool(x4).shape) 
        x5 = self.x1conv64to128(x3) + self.x1conv64to128(x3)
        x6 = self.x1conv128to128(x5) + x5
        # print("15 ", self.pool(x8).shape)
        x9 = self.x1conv128to256(x6) + self.x1conv128to256(x6)
        x10 = self.x1conv256to256(x9) + x9
        # print("3", x10.shape)
        x15 = self.x1conv256to512(x10) + self.x1conv256to512(x10)
        x16 = self.x1conv512to512(x15) + x15
        # print("4 ", x15.shape)
        x18 = self.x1conv512to1024(x16) + self.x1conv512to1024(x16)
        # print("5 ", x16.shape)
        y = self.down(x18)
        # print(y.shape)
        return y

class resNet34(nn.Module):
            
    def __init__(self, in_c, out_c):
        super(resNet34, self).__init__() 
        self.out_c = out_c
        self.x7conv3to64 = nn.Conv2d(in_c,64,7,1,3,bias=False)
        self.x1conv64to128 = nn.Conv2d(64,128,1,1,bias=False)
        self.x1conv128to256 = nn.Conv2d(128,256,1,1,bias=False)
        self.x1conv256to512 = nn.Conv2d(256,512,1,1,bias=False)
        self.x1conv512to1028 = nn.Conv2d(512,1028,1,1,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.x1conv64to64 = DoubleConv(64,64)
        self.x1conv64to128 = Down(64,128)
        self.x1conv128to128 = DoubleConv(128,128)
        self.x1conv128to256 = Down(128,256)
        self.x1conv256to256 = DoubleConv(256,256)
        self.x1conv256to512 = Down(256,512)
        self.x1conv512to512 = DoubleConv(512,512)
        self.x1conv512to1024 = Down(512,1024)
        self.down = resdown(1024, out_c)
    def forward(self, x):
        x1 = self.pool(self.x7conv3to64(x))
        # print("0 ", self.pool(x1).shape) 
        x2 = self.x1conv64to64(x1) + x1
        x3 = self.x1conv64to64(x2) + x2
        x4 = self.x1conv64to64(x2) + x3
        # print("1 ", self.pool(x4).shape) 
        x5 = self.x1conv64to128(x4) + self.x1conv64to128(x4)
        x6 = self.x1conv128to128(x5) + x5
        x7 = self.x1conv128to128(x6) + x6
        x8 = self.x1conv128to128(x7) + x7
        # print("15 ", self.pool(x8).shape)
        x9 = self.x1conv128to256(x8) + self.x1conv128to256(x8)
        x10 = self.x1conv256to256(x9) + x9
        x11 = self.x1conv256to256(x10) + x10
        x12 = self.x1conv256to256(x11) + x11
        x13 = self.x1conv256to256(x12) + x12
        x14 = self.x1conv256to256(x13) + x13
        # print("3", x10.shape)
        x15 = self.x1conv256to512(x14) + self.x1conv256to512(x14)
        x16 = self.x1conv512to512(x15) + x15
        x17 = self.x1conv512to512(x16) + x16
        # print("4 ", x15.shape)
        x18 = self.x1conv512to1024(x17) + self.x1conv512to1024(x17)
        # print("5 ", x16.shape)
        y = self.down(x18)
        # print(y.shape)
        return y

class tribleConv(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(tribleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel*4, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class resNet50(nn.Module):
            
    def __init__(self, in_c, out_c):
        super(resNet50, self).__init__() 
        self.out_c = out_c
        self.x7conv3to64 = nn.Conv2d(in_c,64,7,1,3,bias=False)
        self.x1conv64to128 = nn.Conv2d(64,128,1,1,bias=False)
        self.x1conv128to256 = nn.Conv2d(128,256,1,1,bias=False)
        self.x1conv256to512 = nn.Conv2d(256,512,1,1,bias=False)
        self.x1conv512to1028 = nn.Conv2d(512,1028,1,1,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.x1conv64to64 = tribleConv(64,64)
        self.x1conv64to128 = Down(64,128)
        self.x1conv128to128 = tribleConv(128,128)
        self.x1conv128to256 = Down(128,256)
        self.x1conv256to256 = tribleConv(256,256)
        self.x1conv256to512 = Down(256,512)
        self.x1conv512to512 = tribleConv(512,512)
        self.x1conv512to1024 = Down(512,1024)
        self.down = resdown(1024, out_c)
    def forward(self, x):
        x1 = self.pool(self.x7conv3to64(x))
        # print("0 ", self.pool(x1).shape) 
        x2 = self.x1conv64to64(x1) + x1
        x3 = self.x1conv64to64(x2) + x2
        x4 = self.x1conv64to64(x2) + x3
        # print("1 ", self.pool(x4).shape) 
        x5 = self.x1conv64to128(x4) + self.x1conv64to128(x4)
        x6 = self.x1conv128to128(x5) + x5
        x7 = self.x1conv128to128(x6) + x6
        x8 = self.x1conv128to128(x7) + x7
        x88 = self.x1conv128to128(x8) + x8
        # print("15 ", self.pool(x8).shape)
        x9 = self.x1conv128to256(x88) + self.x1conv128to256(x88)
        x10 = self.x1conv256to256(x9) + x9
        x11 = self.x1conv256to256(x10) + x10
        x12 = self.x1conv256to256(x11) + x11
        x13 = self.x1conv256to256(x12) + x12
        x14 = self.x1conv256to256(x13) + x13
        x144 = self.x1conv256to256(x14) + x14
        # print("3", x10.shape)
        x15 = self.x1conv256to512(x144) + self.x1conv256to512(x144)
        x16 = self.x1conv512to512(x15) + x15
        x17 = self.x1conv512to512(x16) + x16
        x177 = self.x1conv512to512(x16) + x16
        # print("4 ", x15.shape)
        x18 = self.x1conv512to1024(x177) + self.x1conv512to1024(x177)
        # print("5 ", x16.shape)
        y = self.down(x18)
        # print(y.shape)
        return y

class resNet101(nn.Module):
            
    def __init__(self, in_c, out_c):
        super(resNet101, self).__init__() 
        self.out_c = out_c
        self.x7conv3to64 = nn.Conv2d(in_c,64,7,1,3,bias=False)
        self.x1conv64to128 = nn.Conv2d(64,128,1,1,bias=False)
        self.x1conv128to256 = nn.Conv2d(128,256,1,1,bias=False)
        self.x1conv256to512 = nn.Conv2d(256,512,1,1,bias=False)
        self.x1conv512to1028 = nn.Conv2d(512,1028,1,1,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.x1conv64to64 = tribleConv(64,64)
        self.x1conv64to128 = Down(64,128)
        self.x1conv128to128 = tribleConv(128,128)
        self.x1conv128to256 = Down(128,256)
        self.x1conv256to256 = tribleConv(256,256)
        self.x1conv256to512 = Down(256,512)
        self.x1conv512to512 = tribleConv(512,512)
        self.x1conv512to1024 = Down(512,1024)
        self.down = resdown(1024, out_c)
    def forward(self, x):
        x1 = self.pool(self.x7conv3to64(x))
        # print("0 ", self.pool(x1).shape) 
        x2 = self.x1conv64to64(x1) + x1
        x3 = self.x1conv64to64(x2) + x2
        x4 = self.x1conv64to64(x2) + x3
        # print("1 ", self.pool(x4).shape) 
        x5 = self.x1conv64to128(x4) + self.x1conv64to128(x4)
        x6 = self.x1conv128to128(x5) + x5
        x7 = self.x1conv128to128(x6) + x6
        x8 = self.x1conv128to128(x7) + x7
        # print("15 ", self.pool(x8).shape)
        x9 = self.x1conv128to256(x8) + self.x1conv128to256(x8)
        for i in range(23):
            x9 = self.x1conv256to256(x9) + x9
        # print("3", x10.shape)
        x15 = self.x1conv256to512(x9) + self.x1conv256to512(x9)
        x16 = self.x1conv512to512(x15) + x15
        x17 = self.x1conv512to512(x16) + x16
        # print("4 ", x15.shape)
        x18 = self.x1conv512to1024(x17) + self.x1conv512to1024(x17)
        # print("5 ", x16.shape)
        y = self.down(x18)
        # print(y.shape)
        return y
    
class resNet152(nn.Module):
            
    def __init__(self, in_c, out_c):
        super(resNet152, self).__init__() 
        self.out_c = out_c
        self.x7conv3to64 = nn.Conv2d(in_c,64,7,1,3,bias=False)
        self.x1conv64to128 = nn.Conv2d(64,128,1,1,bias=False)
        self.x1conv128to256 = nn.Conv2d(128,256,1,1,bias=False)
        self.x1conv256to512 = nn.Conv2d(256,512,1,1,bias=False)
        self.x1conv512to1028 = nn.Conv2d(512,1028,1,1,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.x1conv64to64 = tribleConv(64,64)
        self.x1conv64to128 = Down(64,128)
        self.x1conv128to128 = tribleConv(128,128)
        self.x1conv128to256 = Down(128,256)
        self.x1conv256to256 = tribleConv(256,256)
        self.x1conv256to512 = Down(256,512)
        self.x1conv512to512 = tribleConv(512,512)
        self.x1conv512to1024 = Down(512,1024)
        self.down = resdown(1024, out_c)
    def forward(self, x):
        x1 = self.pool(self.x7conv3to64(x))
        # print("0 ", self.pool(x1).shape) 
        for i in range(3):
            x1 = self.x1conv64to64(x1) + x1
        # print("1 ", self.pool(x1).shape) 
        x5 = self.x1conv64to128(x1) + self.x1conv64to128(x1)
        for i in range(8):
            x5 = self.x1conv128to128(x5) + x5
        # print("15 ", self.pool(x5).shape)
        x9 = self.x1conv128to256(x5) + self.x1conv128to256(x5)
        for i in range(36):
            x9 = self.x1conv256to256(x9) + x9
        # print("3", x9.shape)
        x15 = self.x1conv256to512(x9) + self.x1conv256to512(x9)
        for i in range(3):
            x15 = self.x1conv512to512(x15) + x15
        # print("4 ", x15.shape)
        x18 = self.x1conv512to1024(x15) + self.x1conv512to1024(x15)
        # print("5 ", x18.shape)
        y = self.down(x18)
        # print(y.shape)
        return y







if __name__ == "__main__":
    x = torch.randn(10, 1, 224, 224)
    res = resNet152(1, 2)
    y = res(x)

    print(f"input shape: {x.shape}")
    print(f"input shape: {y.shape}")
















