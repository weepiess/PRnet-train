import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

'''
def resBlock(x, num_outputs, kernel_size = 4, stride=1):
    assert num_outputs%2==0 #num_outputs must be divided by channel_factor(2 here)
    shortcut = x
    if stride != 1 or x.size()[1] != num_outputs:
        A = nn.Conv2d(in_channels = x.size()[2],out_channels = num_outputs,kernel_size = 1,stride = stride)
        shortcut = A(x)
    B = nn.Sequential(
            nn.Conv2d(in_channels=num_outputs,out_channels=num_outputs/2,kernel_size=1,stride=1),
            nn.BatchNorm2d(num_outputs/2),
            nn.ReLU(True),
            nn.functional.pad,
            nn.Conv2d(in_channels=num_outputs/2,out_channels=num_outputs/2,kernel_size=kernel_size,stride=stride,
                      padding=math.ceil((kernel_size-stride)/2)),
            nn.BatchNorm2d(num_outputs/2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_outputs/2,out_channels=num_outputs,kernel_size=1,stride=1)
        )
    x = B(x)

    x += shortcut
    C = nn.Sequential(
        nn.BatchNorm2d(num_outputs),
        nn.ReLU()
    )
    x = C(x)

    return x
'''
'''
class resfcn256(nn.Module):
    def __init__(self, resolution_inp = 256, resolution_op = 256, channel = 3, name = 'resfcn256'):
        super(resfcn256, self).__init__()
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
    def forward(self, x):
        size = 16
        A = nn.Sequential(
            nn.Conv2d(in_channels=x.size[1], out_channels=size, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size),
            nn.ReLU(True)
        )
        se = A(x)
        se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
        se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
        se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
        se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
        se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
        se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
        se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
        se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
        se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
        se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512
        B = nn.Sequential(
            nn.ConvTranspose2d(in_channels=se.size[1], out_channels=size * 32, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 32),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 32, out_channels=size * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 16, out_channels=size * 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 16, out_channels=size * 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 16, out_channels=size * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 8, out_channels=size * 8, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 8, out_channels=size * 8, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 8, out_channels=size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 4, out_channels=size * 4, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 4, out_channels=size * 4, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 4, out_channels=size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 2, out_channels=size * 2, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 2, out_channels=size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size, out_channels=size, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size, out_channels=3, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        pd = B(se)
        C = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
        pos = C(pd)
        return pos
'''


class resfcn256(nn.Module):
    def __init__(self):
        super(resfcn256, self).__init__()
        self.A = nn.Sequential(
            nn.Conv2d(3, out_channels=16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.B = nn.Sequential(
            nn.ConvTranspose2d(in_channels=se.size[1], out_channels=size * 32, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 32),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 32, out_channels=size * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 16, out_channels=size * 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 16, out_channels=size * 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 16, out_channels=size * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 8, out_channels=size * 8, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 8, out_channels=size * 8, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 8, out_channels=size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 4, out_channels=size * 4, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 4, out_channels=size * 4, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 4, out_channels=size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 2, out_channels=size * 2, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size * 2, out_channels=size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size, out_channels=size, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=size, out_channels=3, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.C = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(3)
        )
        self.ra = nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 1,stride = 2)
        self.rc = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 1,stride = 2)
        self.re = nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 1,stride = 2)
        self.rg = nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 1,stride = 2)
        self.ri = nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 1,stride = 2)
        self.rA = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=1,stride=1)
        )
        self.rB = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=4,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=1,stride=1)
        )
        self.rC = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1)
        )
        self.rD = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1)
        )
        self.rE = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=1)
        )
        self.rF = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=1)
        )
        self.rG = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=1)
        )
        self.rH = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=1)
        )
        self.rI = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1)
        )
        self.rJ = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=4,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1)
        )
        self.r1 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(True))
        self.r2 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True))
        self.r3 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True))
        self.r4 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True))
        self.r5 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))

    def forward(self, x):
        se = self.A(x)

        se_r = self.ra(se)
        se = self.rA(se_r)
        se += se_r
        se = self.r1(se)

        se_r = se
        se = self.rB(se_r)
        se += se_r
        se = self.r1(se)

        se_r = self.rc(se)
        se = self.rC(se_r)
        se += se_r
        se = self.r2(se)

        se_r = se
        se = self.rD(se_r)
        se += se_r
        se = self.r2(se)

        se_r = self.re(se)
        se = self.rE(se_r)
        se += se_r
        se = self.r3(se)

        se_r = se
        se = self.rF(se_r)
        se += se_r
        se = self.r3(se)

        se_r = self.rg(se)
        se = self.rG(se_r)
        se += se_r
        se = self.r4(se)

        se_r = se
        se = self.rH(se_r)
        se += se_r
        se = self.r4(se)

        se_r = self.ri(se)
        se = self.rI(se_r)
        se += se_r
        se = self.r5(se)

        se_r = se
        se = self.rJ(se_r)
        se += se_r
        se = self.r5(se)

        pd = self.B(se)
        pos = self.C(pd)
        pos = F.sigmoid(pos)
        return pos


