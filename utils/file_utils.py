import os, sys, shutil
import torch
import glob

def rm_sub_files(path):
    shutil.rmtree(path)
    os.mkdir(path)

def load_checkpoint(path='./models'):

    files = glob.glob(os.path.join(path, '*.pth.tar'))
    files.sort(reverse=True)
    ckpt_file = files[0]
    print('loading ok!')

    return torch.load(ckpt_file)

def save_checkpoint(state, globel_iter, path='./models', max_keep=10):
    filename = os.path.join(path, '{:06d}.pth.tar'.format(globel_iter))
    torch.save(state, filename)

    files = sorted(os.listdir(path))
    rm_files = files[0: max(0, len(files)-max_keep)]
    for f in rm_files:
        os.remove(os.path.join(path, f))