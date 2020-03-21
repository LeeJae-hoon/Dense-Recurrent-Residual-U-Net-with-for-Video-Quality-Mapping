import torch
import torch.nn as nn
import numpy as np


class DRCL_A(nn.Module):
    def __init__(self, ch_out, t=3):
        super(DRCL_A, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t + 1):
            if i == 0:
                new = self.conv(x)
                sum = x + new
            else:
                new = self.conv(sum)
                sum = sum + new

        return new


# no relu at last
class DRCL_A2(nn.Module):
    def __init__(self, ch_out, t=3):
        super(DRCL_A2, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        for i in range(self.t + 1):
            if i == 0:
                new = self.conv(x)
                new = self.relu(new)
                sum = x + new

            elif i == self.t:
                new = self.conv(sum)

            else:
                new = self.conv(sum)
                new = self.relu(new)
                sum = sum + new

        return new


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=3):
        super(RRCNN_block,self).__init__()
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(
            DRCL_A(ch_out, t=t),
            DRCL_A(ch_out, t=t)
        )

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.RCNN(x)
        return x+x1


# no relu at last
class RRCNN_block2(nn.Module):
    def __init__(self,ch_in,ch_out,t=3):
        super(RRCNN_block2,self).__init__()
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(
            DRCL_A(ch_out, t=t),
            DRCL_A2(ch_out, t=t)
        )

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class MU_block(nn.Module):
    def __init__(self, img_ch=9, output_ch=3, t=3):
        super(R2U_Net, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=img_ch, out_channels=120, kernel_size=3, stride=1, padding=1)
        self.RRCNN1 = RRCNN_block(ch_in=120, ch_out=56, t=t)

        self.Conv2 = nn.Conv2d(in_channels=56, out_channels=112, kernel_size=3, stride=2, padding=1)
        self.RRCNN2 = RRCNN_block(ch_in=112, ch_out=112, t=t)

        self.Conv3 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=3, stride=2, padding=1)
        self.RRCNN3 = RRCNN_block(ch_in=224, ch_out=224, t=t)

        self.Conv4 = nn.Conv2d(in_channels=224, out_channels=448, kernel_size=3, stride=2, padding=1)
        self.RRCNN4 = RRCNN_block2(ch_in=448, ch_out=896, t=t)


        self.Up4 = nn.PixelShuffle(2)
        self.Att4 = Attention_block(F_g=224, F_l=224, F_int=112)
        self.UpConv4 = nn.Conv2d(in_channels=448, out_channels=448, kernel_size=3, stride=1, padding=1)
        self.Up_RRCNN4 = RRCNN_block2(ch_in=448, ch_out=448, t=t)

        self.Up3 = nn.PixelShuffle(2)
        self.Att3 = Attention_block(F_g=112, F_l=112, F_int=56)
        self.UpConv3 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1)
        self.Up_RRCNN3 = RRCNN_block2(ch_in=224, ch_out=224, t=t)

        self.Up2 = nn.PixelShuffle(2)
        self.Att2 = Attention_block(F_g=56, F_l=56, F_int=28)
        self.UpConv2 = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1)   #삭제 가능.
        self.Up_RRCNN2 = RRCNN_block(ch_in=112, ch_out=64, t=t)

        self.Conv_3x3 = nn.Conv2d(64, output_ch, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, ref):

        # encoding path
        x1 = self.Conv1(x)
        x1 = self.relu(x1)
        x1 = self.RRCNN1(x1)

        x2 = self.Conv2(x1)
        x2 = self.relu(x2)
        x2 = self.RRCNN2(x2)

        x3 = self.Conv3(x2)
        x3 = self.relu(x3)
        x3 = self.RRCNN3(x3)

        x4 = self.Conv4(x3)
        x4 = self.relu(x4)
        x4 = self.RRCNN4(x4)

        # decoding path
        y4 = self.Up4(x4)
        x3 = self.Att4(g=y4, x=x3)
        y4 = torch.cat((x3, y4), dim=1)
        y4 = self.UpConv4(y4)
        y4 = self.relu(y4)
        y4 = self.Up_RRCNN4(y4)

        y3 = self.Up3(y4)
        x2 = self.Att3(g=y3, x=x2)
        y3 = torch.cat((x2, y3), dim=1)
        y3 = self.UpConv3(y3)
        y3 = self.relu(y3)
        y3 = self.Up_RRCNN3(y3)

        y2 = self.Up2(y3)
        x1 = self.Att2(g=y2, x=x1)
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.UpConv2(y2)
        y2 = self.Up_RRCNN2(y2)

        y1 = self.Conv_3x3(y2)

        #residual learning
        re = ref - y1
        return re

