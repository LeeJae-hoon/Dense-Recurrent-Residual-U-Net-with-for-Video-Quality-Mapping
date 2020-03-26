import torch
import torch.nn as nn
import numpy as np



class DRCL_C(nn.Module):
    def __init__(self, ch_out, t=3):
        super(DRCL_C, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(2*ch_out, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3*ch_out, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(4*ch_out, ch_out, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.conv(x)

        x2 = torch.cat((x, x1), dim=1)
        x3 = self.conv1(x2)
        x3 = self.conv(x3)

        x4 = torch.cat((x2, x3), dim=1)
        x5 = self.conv2(x4)
        x5 = self.conv(x5)

        x6 = torch.cat((x4, x5), dim=1)
        x7 = self.conv3(x6)
        x7 = self.conv(x7)

        return x7


# no ReLU after the last convolution
class DRCL_C2(nn.Module):
    def __init__(self, ch_out, t=3):
        super(DRCL_C2, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(2*ch_out, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3*ch_out, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(4*ch_out, ch_out, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(x1)

        x2 = torch.cat((x, x1), dim=1)
        x3 = self.conv1(x2)
        x3 = self.conv(x3)
        x3 = self.relu(x3)

        x4 = torch.cat((x2, x3), dim=1)
        x5 = self.conv2(x4)
        x5 = self.conv(x5)
        x5 = self.relu(x5)

        x6 = torch.cat((x4, x5), dim=1)
        x7 = self.conv3(x6)
        x7 = self.conv(x7)

        return x7



class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=3):
        super(RRCNN_block,self).__init__()
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(
            DRCL_C(ch_out, t=t),
            DRCL_C(ch_out, t=t)
        )

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.RCNN(x)
        return x+x1


# no ReLU after the last convolution
class RRCNN_block2(nn.Module):
    def __init__(self,ch_in,ch_out,t=3):
        super(RRCNN_block2,self).__init__()
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(
            DRCL_C(ch_out, t=t),
            DRCL_C2(ch_out, t=t)
        )

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class MU_block(nn.Module):
    def __init__(self, img_ch=9, output_ch=3, t=3):
        super(R2U_Net, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=img_ch, out_channels=90, kernel_size=3, stride=1, padding=1)
        self.RRCNN1 = RRCNN_block(ch_in=90, ch_out=32, t=t)

        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=64, t=t)

        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.RRCNN3 = RRCNN_block2(ch_in=128, ch_out=256, t=t)


        self.Up3 = nn.PixelShuffle(2)
        self.UpConv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.Up_RRCNN3 = RRCNN_block2(ch_in=128, ch_out=128, t=t)

        self.Up2 = nn.PixelShuffle(2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=64, t=t)

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

        # decoding path
        y3 = self.Up3(x3)
        y3 = torch.cat((x2, y3), dim=1)
        y3 = self.UpConv3(y3)
        y3 = self.relu(y3)
        y3 = self.Up_RRCNN3(y3)

        y2 = self.Up2(y3)
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.Up_RRCNN2(y2)

        y1 = self.Conv_3x3(y2)

        # residual learning
        re = ref - y1
        return re

