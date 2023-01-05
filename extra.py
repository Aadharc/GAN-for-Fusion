import torch

x = torch.randn((1, 3, 640, 512))
y = torch.randn((1, 3, 640, 512))
z1 = torch.cat([x,y], dim = 2)
z2 = torch.cat([x,y], dim = 1)

print(z1.shape)
print(z2.shape)