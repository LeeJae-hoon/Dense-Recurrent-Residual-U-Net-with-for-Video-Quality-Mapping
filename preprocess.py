import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import numpy as np
from PIL import Image
import random


class Video_Provider(Dataset):

    def __init__(self, im_size, maxnumsample, source_path, target_path, frames=5):
        self.im_size = im_size
        self.frames = frames
        self.trans = transforms.ToTensor()
        self.maxnumsample = maxnumsample
        self.source_path = source_path
        self.target_path = target_path

    def __getitem__(self, index):

        if not self.im_size is None:

            base = 'F:/dataset_VQM/training/'

            while True:
                numseq = random.randint(0, 59)
                # exclude "001.mp4" for self-validation during training
                if numseq not in [1]:
                    break

            sourcepath = base + 'source/seq' + str(numseq).rjust(3, '0')
            targetpath = base + 'target/seq' + str(numseq).rjust(3, '0')

            lenfr = len(os.walk(targetpath).__next__()[2])
            midfr = random.randint(2, lenfr - 3)

            sample = Image.open(sourcepath+'/0.png')
            width, height = sample.size

            hs = random.randint(0, height - self.im_size)
            ws = random.randint(0, width - self.im_size)

            source = torch.zeros(self.frames, 3, self.im_size, self.im_size)
            target = torch.zeros(3, self.im_size, self.im_size)

            for i in range(self.frames):
                simg = Image.open(sourcepath+'/'+str(midfr-2+i)+'.png')
                simg = self.trans(simg)
                simg = simg[:, hs:hs+self.im_size, ws:ws+self.im_size]
                try:
                    source[i, :, :, :] = simg
                except:
                    print('Error Occurred at')
                    print('numseq:', numseq, 'lenfr:', lenfr, 'midfr:', midfr, 'hs:', hs, 'ws:', ws)
                    print()

                if i==self.frames//2:
                    timg = Image.open(targetpath+'/'+str(midfr)+'.png')
                    timg = self.trans(timg)[:, hs:hs+self.im_size, ws:ws+self.im_size]
                    target = timg

        return source, target

    def __len__(self):
        return self.maxnumsample



















