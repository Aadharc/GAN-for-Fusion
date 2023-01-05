import numpy as np
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_chan, out_chan, act = "relu") :
        super(Block , self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, padding = 1, bias = True, padding_mode = "reflect"),
            nn.BatchNorm2d(out_chan),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x) 



class Features(nn.Module):
    def __init__(self, in_chan = 3, features = 8 ):
        super(Features, self).__init__()
        self.features1 = Block(in_chan, features, act="relu")
        self.features2 = Block(features, features * 2, act= "relu")
        self.features3 = Block(features*2, features * 4, act= "relu")
        self.features4 = Block(features*4, features * 8, act= "relu")
    def forward(self, x):
        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)
        f4 = self.features4(f3)
        return f4 

class Masks(nn.Module):
    def __init__(self, in_chan = 3, features = 8 ):
        super(Masks, self).__init__()
        self.mask1 = Block(in_chan, features, act="relu")
        self.mask2 = Block(features, features * 2, act= "relu")
        self.mask3 = Block(features*2, features * 4, act= "relu")
        self.mask4 = Block(features*4, features * 8, act= "relu")
    def forward(self, x):
        m1 = self.mask1(x)
        m2 = self.mask2(m1)
        m3 = self.mask3(m2)
        m4 = self.mask4(m3)
        return m4

class MaskedFeatures(nn.Module):
    def __init__(self, in_chan, features) :
        super(MaskedFeatures, self).__init__()
        self.feature = Features(in_chan =3, features = 8)
        self.mask = Masks(in_chan =3, features = 8)
    
    def forward(self, x, y):
        featx = self.feature(x)
        maskx = self.mask(x)
        masked_featx = maskx * featx

        featy = self.feature(y)
        masky = self.mask(y)
        masked_featy = masky * featy

        return [featx, featy, masked_featx, masked_featy]


def test():
    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 3, 512, 512)
    model = MaskedFeatures(in_chan=3, features= 8)
    masked_featx, masked_featy = model(x,y)[2], model(x,y)[3]
    print(model)
    print(masked_featx.shape, masked_featy.shape)

if __name__ == "__main__":
    test()