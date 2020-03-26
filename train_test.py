import torch
import torch.nn as nn
import numpy as np
from arch.Network import Network
from utils.data_utils import *
from utils.file_utils import *
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from preprocess import Video_Provider
import os, sys, shutil
import torch.optim as optim
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import math
from PIL import Image


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='Video Quality Mapping', type=str)
    parser.add_argument('--eval', '-e', default='', help='whether to work on the eval mode')
    parser.add_argument('--training_source', '-dp', default='F:\\dataset_VQM\\training\\source', help='the path of vimeo-90k')
    parser.add_argument('--training_target', '-dp', default='F:\\dataset_VQM\\training\\target', help='the path of vimeo-90k')
    parser.add_argument('--valset_source', default='F:\\dataset_VQM\\training\\source', help='the path of validation source set')
    parser.add_argument('--valset_target', default='F:\\dataset_VQM\\training\\target', help='the path of validation target set')
    parser.add_argument('--test_source', default='F:/dataset_VQM/test', help='the path of validation source set')
    parser.add_argument('--save_path', default='F:/VQM_results', help='the path of validation source set')
    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='batch size')
    parser.add_argument('--frames', '-f', default=5, type=int)
    parser.add_argument('--im_size', '-s', default=96, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--num_worker', '-nw', default=4, type=int, help='number of workers to load data by dataloader')
    parser.add_argument('--restart', '-r', default='True', type=bool, help='whether to restart the train process')
    parser.add_argument('--cuda', default='True', help='whether to train the network on the GPU, default is mGPU')
    parser.add_argument('--max_numsample', default=40000, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    return parser.parse_args()

def lradjust(epoch):

    if epoch>18:
        learningrate2 = 1e-7
    elif epoch>14:
        learningrate2 = 1e-6
    elif epoch>10:
        learningrate2 = 1e-5
    else:
        learningrate2 = 1e-4

    return learningrate2

def train(args):

    data_set = Video_Provider(
        im_size=96,
        frames=5,
        maxnumsample=args.max_numsample,
        source_path=args.training_source,
        target_path=args.training_target
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker
    )

    model = Network(in_frames=args.frames)

    if args.cuda:
        model = nn.DataParallel(model.cuda())
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # b: args.batch_size, N: args.frames, c: channel, h: height, w: width
    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.restart:
        #rm_sub_files('./models')
        epoch = 0
        global_iter = 0
        best_loss = np.inf
        print('Start the train process.')
    else:
        print('Continue the training.')
        state = load_checkpoint('./models')
        epoch = state['epoch']
        global_iter = state['global_iter']
        best_loss = state['best_loss']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['state_dict'])
        print('Model load OK at global_iter {}, epoch {}.'.format(global_iter, epoch))

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    log_writer = SummaryWriter('./logs')

    loss_func = nn.MSELoss()
    trans_PIL = transforms.ToPILImage()
    trans_tensor = transforms.ToTensor()
    model.train()
    lr = args.learning_rate

    for e in range(epoch, args.max_epoch):
        lr = lradjust(e)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        loss_train, psnr_train, ssim_train = 0, 0, 0

        for iter, (data1, gt1) in enumerate(data_loader):
            if args.cuda:
                data1 = data1.cuda()
                gt1 = gt1.cuda()

            pred = model(data1)

            loss = loss_func(gt1, pred)
            global_iter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = calculate_psnr(pred, gt1)
            ssim = calculate_ssim(pred, gt1)

            psnr_train += psnr
            ssim_train += ssim
            loss_train += loss.item()


            if (iter+1) % 1000 == 0:
                loss_train /= 1000
                psnr_train /= 1000
                ssim_train /= 1000
                best_loss = min(best_loss, loss_train)

                print('{:3d} epoch, {:4d} / {} iter, loss: {:.6f}, PSNR: {:.4f}dB, SSIM: {:.4f}'.format(e, iter+1, int(args.max_numsample/args.batch_size), loss_train, psnr_train, ssim_train))

                log_writer.add_scalar('train_loss', loss_train, global_iter)
                log_writer.add_scalar('train_psnr', psnr_train, global_iter)
                log_writer.add_scalar('train_ssim', ssim_train, global_iter)
                state = {
                    'state_dict': model.state_dict(),
                    'epoch': e,
                    'global_iter': global_iter,
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }

                save_checkpoint(state, global_iter, path='./models', max_keep=3)

                loss_train, psnr_train, ssim_train = 0, 0, 0

            # Validation
            if (e > 5 and (iter+1)%10000 == 0) or (e==14 and (iter+1)%5000 == 0):
                print('validation starts')
                with torch.no_grad():
                    model.eval()
                    psnr_sum, fr_sum = 0, 0
                    cuda = args.cuda
                    if cuda:
                        print("CUDA True and Validation starts")
                    if not os.path.exists('./val_images'):
                        os.mkdir('./val_images')
                    else:
                        rm_sub_files('./val_images')

                    seqlist = [1]
                    keepfr = torch.zeros(1, 5, 3, 1080, 1920)
                    targetfr = torch.zeros(1, 3, 1080, 1920)
                    cu_keepfr = torch.zeros(1, 5, 3, 360, 640)
                    cu_targetfr = torch.zeros(1, 3, 360, 640)
                    totalmse = 0

                    for seqidx in seqlist:

                        seqpath = args.valset_source+'/seq'+str(seqidx).rjust(3,'0')
                        targetpath = args.valset_target+'/seq'+str(seqidx).rjust(3, '0')
                        numfr = len(os.walk(seqpath).__next__()[2])
                        if numfr>120:
                            numfr = 120

                        fr_sum += (numfr/2)

                        for fridx in range(0, numfr, 2):
                            if fridx==0:
                                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath+'/2.png'))
                                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath+'/1.png'))
                                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath+'/0.png'))
                                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath+'/1.png'))
                                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath+'/2.png'))

                            elif fridx==1:
                                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath+'/1.png'))
                                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath+'/0.png'))
                                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath+'/1.png'))
                                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath+'/2.png'))
                                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath+'/3.png'))

                            elif fridx==numfr-2:
                                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-4)+'.png'))
                                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-3)+'.png'))
                                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-2)+'.png'))
                                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-1)+'.png'))
                                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-2)+'.png'))

                            elif fridx==numfr-1:
                                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-3)+'.png'))
                                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-2)+'.png'))
                                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-1)+'.png'))
                                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-2)+'.png'))
                                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath+'/'+str(numfr-3)+'.png'))

                            else:
                                keepfr[0, 0, ...] = trans_tensor(Image.open(seqpath+'/'+str(fridx-2)+'.png'))
                                keepfr[0, 1, ...] = trans_tensor(Image.open(seqpath+'/'+str(fridx-1)+'.png'))
                                keepfr[0, 2, ...] = trans_tensor(Image.open(seqpath+'/'+str(fridx)+'.png'))
                                keepfr[0, 3, ...] = trans_tensor(Image.open(seqpath+'/'+str(fridx+1)+'.png'))
                                keepfr[0, 4, ...] = trans_tensor(Image.open(seqpath+'/'+str(fridx+2)+'.png'))

                            targetfr[0, ...] = trans_tensor(Image.open(targetpath+'/'+str(fridx)+'.png'))

                            for hidx in range(3):
                                for widx in range(3):
                                    cu_keepfr[0, 0, ...] = keepfr[0, 0, ...][:, 360*hidx:360*hidx+360, 640*widx:640*widx+640]
                                    cu_keepfr[0, 1, ...] = keepfr[0, 1, ...][:, 360*hidx:360*hidx+360, 640*widx:640*widx+640]
                                    cu_keepfr[0, 2, ...] = keepfr[0, 2, ...][:, 360*hidx:360*hidx+360, 640*widx:640*widx+640]
                                    cu_keepfr[0, 3, ...] = keepfr[0, 3, ...][:, 360*hidx:360*hidx+360, 640*widx:640*widx+640]
                                    cu_keepfr[0, 4, ...] = keepfr[0, 4, ...][:, 360*hidx:360*hidx+360, 640*widx:640*widx+640]

                                    cu_targetfr[0, ...] = targetfr[0, ...][:, 360*hidx:360*hidx+360, 640*widx:640*widx+640]

                                    cu_keepfr = cu_keepfr.cuda()
                                    cu_targetfr = cu_targetfr.cuda()

                                    val_pred = model(cu_keepfr)

                                    val_pred.data = val_pred.data.clamp(0., 1.)

                                    totalmse += calculate_mse(val_pred, cu_targetfr)*360*640

                            totalmse /= 1080*1920
                            psnr_val = 10*math.log10(255*255/totalmse)

                            psnr_sum += psnr_val

                            val_pred = val_pred.cpu()

                            #if fridx < 50:
                            #    trans_PIL(val_pred[0,:,:,:]).save('./val_images/{}_{}_pred_{:.2f}dB.png'.format(seqidx, fridx, psnr_val, quality=95))

                    psnr_sum = psnr_sum/fr_sum

                    print()
                    print('*******************************************************************************************')
                    print('{} global iter. validation results'.format(global_iter+1))
                    print('PSNR: {:.2f}'.format(psnr_sum))
                    print('*******************************************************************************************')
                    print()

                    log_writer.add_scalar('val_psnr', psnr_sum, global_iter)

                model.train()


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

    seqlist = list(range(60, 76))
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
    if not args.eval:
        train(args)
    else:
        with torch.no_grad():
            eval(args)