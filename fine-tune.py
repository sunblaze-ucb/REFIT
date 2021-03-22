"""Train CIFAR with PyTorch."""
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass
from helpers.loaders import *
from helpers.utils import adjust_learning_rate
from helpers.utils import re_initializer_layer
from trainer import test, train, test_afs
import models

parser = argparse.ArgumentParser(description='Fine-tune models.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the testing data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to use')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path to the wm set')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the file of the wm random labels under wm_path')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
parser.add_argument('--num_workers', default=4, type=int, help='the number of workers for loaders')

parser.add_argument('--max_epochs', default=20, type=int, help='the maximum number of epochs')
parser.add_argument('--load_path', default='./checkpoint/model.t7', help='the path to the pre-trained model')
parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
parser.add_argument('--save_model', default='finetune.t7', help='model name')
parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='finetune', help='the exp name')
parser.add_argument('--tunealllayers', action='store_true', help='enable to fine-tune all layers otherwise only last layer')
parser.add_argument('--reinitll', action='store_true', help='re initialize the last layer')

parser.add_argument('--optim', type=str, default='SGD', help='type of optimizer utilized')
parser.add_argument('--transfer', action='store_true', help='replace the last layer with the orignal one when verify')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='parameter for SGD optimizer')
parser.add_argument('--momentum', default=0.9, type=float, help='parameter for SGD optimizer')


parser.add_argument('--wm_batch_size', default=100, type=int, help='the wm batch size')
parser.add_argument('--embed', action='store_true', help='watermarking with wm set 2')
parser.add_argument('--wm2_path', default='', help='the path the wm set 2')

parser.add_argument('--wm2_lbl', default='labels-pred.txt', help='the path the wm2 random labels')

parser.add_argument('--lradj', default=99999, type=int, help='multiple the lr by ratio every lradj epochs')
parser.add_argument('--ratio', default=0.1, type=float, help='multiple the lr by ratio every lradj epochs')
parser.add_argument('--period', default=99999, type=int, help='the period which the learning rate is repeated')


parser.add_argument('--extra_data', default='', help='extra data sources in train, seperated by :')
parser.add_argument('--extra_data_bsize', default='', help='batch size for extra data sources, seperated by :')
parser.add_argument('--extra_net', default='', help='the path to the model used to label the unlabeled data')

parser.add_argument('--extra_only', action='store_true', help='when enabled, use only extra data sources in fine-tuning')

parser.add_argument('--model', default='resnet18', help='architecture of the the model')


parser.add_argument('--EWC_coef', default=0., type=float, help='coef for EWC')
parser.add_argument('--EWC_samples', default=1000, type=int, help='samples for approximiating Fisher Infomation')

parser.add_argument('--load_path_private', default='', help='the path to the pre-trained model with private key')

parser.add_argument('--exp_weighting', default=0, type=float, help='hyperparameter T in exponential weight')


parser.add_argument('--wm_afs', action='store_true')
parser.add_argument('--wm_afs_bsize', type=int, default = 0)


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


LOG_DIR = args.log_dir
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'log_' + str(args.runname) + '.txt')
confgfile = os.path.join(LOG_DIR, 'conf_' + str(args.runname) + '.txt')

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

trainloader, testloader, n_classes = getdataloader(
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size, args.num_workers)

# load watermark images
if not args.wm_afs:
    print('Loading watermark images')
    wmloader = getwmloader(args.wm_path, args.batch_size, args.wm_lbl)

if len(args.wm2_path) > 0:
    wm2loader = getwmloader(args.wm2_path, args.wm_batch_size, args.wm2_lbl)
else:
    wm2loader = None

if len(args.extra_net) > 0:
    extra_net = torch.load(args.extra_net)['net']
    if device == 'cuda':
        extra_net = torch.nn.DataParallel(extra_net, device_ids=range(torch.cuda.device_count()))
    extra_net.eval()
else:
    extra_net = None


extra_loaders = []
if len(args.extra_data) > 0:
    ex_data = args.extra_data.split(':')
    ex_batch_size = args.extra_data_bsize.split(':')
    for i in range(len(ex_data)):
        if ex_data[i].split('+')[0] == 'wm':
            tmp = ex_data[i].split('+')
            _loader = getwmloader(tmp[1], int(ex_batch_size[i]), tmp[2])
        else:
            _loader, _, __ = getdataloader(ex_data[i], args.train_db_path, args.test_db_path, int(ex_batch_size[i]), 4)
        extra_loaders.append(batch_gen(_loader))    

# Loading model.
print('==> loading model...')
if args.load_path == 'resnet18':
    net = models.ResNet18(num_classes=n_classes)
else:
    checkpoint = torch.load(args.load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
start_epoch = 0#checkpoint['epoch']

net = net.to(device)
# support cuda
if device == 'cuda':
    print('Using CUDA')
    print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.wm_afs:
    if args.load_path == 'resnet18':
        checkpoint = torch.load(args.extra_net)

    net.afs_inputs = checkpoint['afs_inputs']
    net.afs_targets = checkpoint['afs_targets']

# re initialize and re train the last layer
#if not hasattr(net.module, 'linear'):
#    net.module.linear = nn.Linear(1, 1)

if len(args.load_path_private) > 0:
    checkpoint2 = torch.load(args.load_path_private)
    net2 = checkpoint2['net'].to(device)
    private_key = net2.linear
    
else:
    private_key = net.module.linear


if args.EWC_coef > 0:
    net.eval()
    grad_sum = [param.new_zeros(param.size()) for param in net.parameters()]
    optimizer = optim.SGD(net.parameters(), lr=0.123)#this line is not the optimizer used for actual training!

    sample_cnt = 0
    while True:
        for inputs, targets in trainloader:
            if sample_cnt >= args.EWC_samples:
                continue

            inputs, targets = inputs.to(device), targets.to(device)
        
            prob = F.softmax(net(inputs), dim=1)
        
            lbls = torch.multinomial(prob, 1).to(device)
        
            log_prob = torch.log(prob)
        
            for i in range(inputs.size(0)):
                optimizer.zero_grad()
                log_prob[i][lbls[i]].backward(retain_graph=True)
                with torch.no_grad():
                    grad_sum = [g + (param.grad.data.detach()**2) for g, param in zip(grad_sum, net.parameters())]

            sample_cnt += inputs.size(0)
            print ("Approximating Fisher: %.3f"%(float(sample_cnt) / args.EWC_samples))
        if sample_cnt >= args.EWC_samples:
            break
    
    Fisher = [g / sample_cnt for g in grad_sum]

    

    _fmax = 0
    _fmin = 1e9
    _fmean = 0.
    for g in Fisher:
        _fmax = max(_fmax, g.max())
        _fmin = min(_fmin, g.min())
        _fmean += g.mean()
    print ("[max: %.3f] [min: %.3f] [avg: %.3f]"%(_fmax, _fmin, _fmean / len(Fisher)))

    Fisher = [g / _fmax for g in Fisher]

    init_params = [param.data.clone().detach() for param in net.parameters()]
    
else:
    Fisher = None
    init_params = None

if args.reinitll:
    if args.model == 'resnet18':
        net, _ = re_initializer_layer(net, n_classes)
        EWC_immune = [p for p in net.module.linear.parameters()]
    elif args.model == 'vgg16_bn':
        _ = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )
        net, _ = re_initializer_layer(net, n_classes, _)
        EWC_immune = [p for p in net.module.linear.parameters()]
    else:
        raise Exception("Unsupported args.model")
else:
    EWC_immune = []
try:
    if device is 'cuda':
        net.module.unfreeze_model()
    else:
        net.unfreeze_model()
except:
    print ("no unfreeze_model in net")

if args.exp_weighting != 0:
    for mod in net.modules():
        mod.EW_T = args.exp_weighting

criterion = nn.CrossEntropyLoss()
if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optim == 'RMSprop':
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
# start training loop

new_layer = net.module.linear
if args.transfer:
    net, _ = re_initializer_layer(net, 0, private_key)


print("WM acc:")
if not args.wm_afs:
    test(net, criterion, logfile, wmloader, device)
else:
    test_afs(net, logfile)

if len(args.wm2_path) > 0:
    print("WM2 acc:")
    test(net, criterion, logfile, wm2loader, device)

net, _ = re_initializer_layer(net, 0, new_layer)

print("Test acc:")
test(net, criterion, logfile, testloader, device)


        

# start training
for epoch in range(start_epoch, start_epoch + args.max_epochs):
    adjust_learning_rate(args.lr, optimizer, epoch, args.lradj, args.ratio, args.period)
    if epoch % args.period == 0 and args.reinitll and epoch != 0:
        net, _ = re_initializer_layer(net, n_classes)

    train(epoch, net, criterion, optimizer, logfile,
            trainloader, device, wmloader=wm2loader if args.embed else False, tune_all=args.tunealllayers, ex_datas = extra_loaders, ex_net = extra_net, n_classes=n_classes, EWC_coef = args.EWC_coef, Fisher = Fisher, init_params = init_params, EWC_immune = EWC_immune, afs_bsize = args.wm_afs_bsize if args.wm_afs else 0, extra_only = args.extra_only)

    print("Test acc:")
    acc = test(net, criterion, logfile, testloader, device)

    # replacing the last layer to check the wm resistance
    new_layer = net.module.linear
    if args.transfer:
        net, _ = re_initializer_layer(net, 0, private_key)
    
    print("WM acc:")
    if not args.wm_afs:
        test(net, criterion, logfile, wmloader, device)
    else:
        test_afs(net, logfile)

    if len(args.wm2_path) > 0:
        print("WM2 acc:")
        test(net, criterion, logfile, wm2loader, device)
    
    # plugging the new layer back
    net, _ = re_initializer_layer(net, 0, new_layer)

    print('Saving..')
    state = {
        'net': net.module if device is 'cuda' else net,
        'acc': acc,
        'epoch': epoch,
    }
    if args.wm_afs:
        state['afs_inputs'] = net.afs_inputs 
        state['afs_targets'] = net.afs_targets

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    torch.save(state, os.path.join(args.save_dir, str(args.runname) + str(args.save_model)))
