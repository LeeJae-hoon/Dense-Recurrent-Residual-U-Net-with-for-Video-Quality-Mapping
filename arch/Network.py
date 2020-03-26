import torch
import torch.nn as nn
from arch.MU_block import MU_block
import numpy as np

class Network(nn.Module):
    def __init__(self, in_frames=5):

        super(Network, self).__init__()
        self.in_frames = in_frames
        channel = 3
        in1 = 3 * channel
        self.block1 = MU_block(in1, channel)
        self.block2 = MU_block(in1, channel)

    def forward(self, input):

        frames = input
        # b: args.batch_size, N: args.frames, c: channel, h: height, w: width
        b, N, c, h, w = frames.size()

        data_temp = []

        for i in range(self.in_frames-2):
            data_temp.append(self.block1(frames[:, i:i+3, ...].view(b, -1, h, w), frames[:, i+1, ...]))

        data_temp = torch.cat(data_temp, dim=1)
        return self.block2(data_temp, frames[:, N//2, ...])
