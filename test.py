import torch
import torch.nn as nn
import numpy as np
from arch.Network import Network
from utils.file_utils import *
import argparse
import os
from torchvision.transforms import transforms
from PIL import Image
import time


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_source', default='F:/dataset_VQM/test', help='the path of validation source set')
    parser.add_argument('--save_path', default='F:/VQM_results', help='the path of validation source set')
    parser.add_argument('--frames', '-f', default=5, type=int)
    parser.add_argument('--im_size', '-s', default=96, type=int)
    parser.add_argument('--cuda', default='True', help='whether to train the network on the GPU, default is mGPU')
    return parser.parse_args()


def eval(args):

    trans_PIL = transforms.ToPILImage()
    trans_tensor = transforms.ToTensor()

    model = Network(5)

    if args.cuda:
        model = nn.DataParallel(model.cuda())

    state = load_checkpoint('./models')
    model.load_state_dict(state['state_dict'])
    model.eval()
    print('Model load OK!')

    if not os.path.exists('F:/VQM_results'):
        os.mkdir('F:/VQM_results')
    #rm_sub_files('F:/VQM_results')

    cuda = args.cuda
    if cuda:
        print("CUDA True and test starts")

    seqlist = list(range(76, 92))
    myhei = 540
    mywid = 960
    hext = 40
    wext = 40
    keepfr = torch.zeros(1, 5, 3, 1080, 1920)
    cu_keepfr = torch.zeros(1, 5, 3, myhei+hext, mywid+wext)
    predsheet = torch.zeros(1, 3, 1080, 1920)
    numfr = 120
    runtime = 0
    #framelist = [19, 39, 59, 79, 99, 119]

    for seqidx in seqlist:

        seqpath = args.test_source + '/' + str(seqidx).rjust(3, '0') + '_'

        for fridx in range(numfr):

            start = time.time()

            if fridx == 0:
                # Use mirror padding method for edge frames
                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath + '000002.png'))
                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath + '000001.png'))
                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath + '000000.png'))
                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath + '000001.png'))
                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath + '000002.png'))

            elif fridx == 1:
                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath + '000001.png'))
                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath + '000000.png'))
                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath + '000001.png'))
                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath + '000002.png'))
                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath + '000003.png'))

            elif fridx == numfr - 2:
                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath + str(numfr - 4).rjust(6, '0') + '.png'))
                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath + str(numfr - 3).rjust(6, '0') + '.png'))
                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath + str(numfr - 2).rjust(6, '0') + '.png'))
                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath + str(numfr - 1).rjust(6, '0') + '.png'))
                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath + str(numfr - 2).rjust(6, '0') + '.png'))

            elif fridx == numfr - 1:
                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath + str(numfr - 3).rjust(6, '0') + '.png'))
                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath + str(numfr - 2).rjust(6, '0') + '.png'))
                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath + str(numfr - 1).rjust(6, '0') + '.png'))
                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath + str(numfr - 2).rjust(6, '0') + '.png'))
                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath + str(numfr - 3).rjust(6, '0') + '.png'))

            else:
                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath + str(fridx - 2).rjust(6, '0') + '.png'))
                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath + str(fridx - 1).rjust(6, '0') + '.png'))
                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath + str(fridx).rjust(6, '0') + '.png'))
                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath + str(fridx + 1).rjust(6, '0') + '.png'))
                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath + str(fridx + 2).rjust(6, '0') + '.png'))

            # hidx, widx = 0, 1
            for hidx in range(2):
                for widx in range(2):
                    for fridx2 in range(5):
                        cu_keepfr[0, fridx2, ...] = keepfr[0, fridx2, ...][:, (myhei + hext) * hidx - 2 * hext * hidx:(myhei + hext) * (hidx + 1) - 2 * hext * hidx,
                                                    (mywid + wext) * widx - 2 * wext * widx: (mywid + wext) * (widx + 1) - 2 * wext * widx]

                    cu_keepfr = cu_keepfr.cuda()

                    val_pred = model(cu_keepfr)

                    val_pred.data = val_pred.data.clamp(0., 1.)
                    val_pred = val_pred.cpu()

                    predsheet[0, :, myhei * hidx:myhei * (hidx + 1), mywid * widx:mywid * (widx + 1)] = val_pred[0, :,
                                                                                                        0 + hext * hidx:myhei + hext * hidx,
                                                                                                        0 + wext * widx:mywid + wext * widx]

            end = time.time()
            runtime += (end - start)

            trans_PIL(predsheet[0, :, :, :]).save(
                'F:/VQM_results/{}_{}.png'.format(str(seqidx).rjust(3, '0'), str(fridx).rjust(6, '0'), quality=100))

            print('{}_{} is done'.format(str(seqidx).rjust(3, '0'), str(fridx).rjust(6, '0')))

    runtime /= (len(seqlist)*numfr)
    print('\nEvaluation ends')
    print('avg runtime: '+str(runtime)+'sec')


if __name__ == '__main__':

    args = args_parser()
    print(args)

    with torch.no_grad():
        eval(args)