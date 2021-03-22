from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from helpers.loaders import *
from helpers.utils import adjust_learning_rate
from models import *
from trainer import test, train, test_afs

parser = argparse.ArgumentParser(description='generate watermarks used in Adversarial Frontier Stitching')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on')
parser.add_argument('--batch_size', default=500, type=int, help='the batch size')
parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
parser.add_argument('--load_path', default='./checkpoint/ckpt.t7', help='the path to the target model, to be used with resume flag')

parser.add_argument('--load_path2', default='', help='optional, the path to another model for evaluating watermark accuracy on an independent model')


parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='train', help='the exp name')

parser.add_argument('--model', default='resnet18', help='architecture of the the model')
parser.add_argument('--eps', type=float, default=0, help='the scale of adversarial perturbation in AFS')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

size100 = args.dataset.split('+')[0] == 'pubfig83'

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.exists(args.load_path), 'Error: no checkpoint found!'
checkpoint = torch.load(args.load_path)
net = checkpoint['net']
acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

net = net.to(device)
# support cuda
if device == 'cuda':
    print('Using CUDA')
    print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

trainloader = batch_gen(trainloader)


def gen_adv(net, eps):
    global trainloader
    net.eval()
    inputs, targets = next(trainloader)
    inputs, targets = inputs.to(device), targets.to(device)
    inputs.requires_grad = True
    
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    loss = criterion(outputs, targets)
    grad = torch.autograd.grad(loss, [inputs])[0]

    with torch.no_grad():
        adv_inputs = inputs + eps * grad.sign()
        adv_outputs = net(adv_inputs)
        _, adv_predicted = torch.max(adv_outputs, 1)
    
    cnt_correct = 0
    cnt_fool = 0
    for i in range(inputs.size(0)):
        if predicted[i] == targets[i]:
            cnt_correct += 1
            if adv_predicted[i] != targets[i]:
                cnt_fool += 1
    return float(cnt_fool) / cnt_correct, adv_inputs, targets, predicted, adv_predicted


eps = args.eps

with open(logfile, 'a') as f:
    f.write("[eps: %.3f][fool_rate: %.3f]\n"%(eps, gen_adv(net, eps)[0]))
print ("[eps: %.3f][fool_rate: %.3f]"%(eps, gen_adv(net, eps)[0]))


tot_wm = 100
tot_true_adv = 50
tot_false_adv = 50
afs_inputs = []
afs_targets = []

while max(tot_true_adv, tot_false_adv) > 0:
    fool_rate, adv_inputs, targets, predicted, adv_predicted = gen_adv(net, eps)
    cnt_true = 0
    cnt_false = 0

    for i in range(adv_inputs.size(0)):
        if predicted[i] == targets[i]:  
            if adv_predicted[i] == targets[i]:
                if tot_false_adv > 0:
                    tot_false_adv -= 1
                    cnt_false += 1
                    afs_inputs.append(adv_inputs[i])
                    afs_targets.append(targets[i])
            else:
                if tot_true_adv > 0:
                    tot_true_adv -= 1
                    cnt_true += 1
                    afs_inputs.append(adv_inputs[i])
                    afs_targets.append(targets[i])
    with open(logfile, 'a') as f:
        f.write("[eps: %.3f][fool_rate: %.3f][true_adv: %d][false_adv: %d]\n"%(eps, fool_rate, cnt_true, cnt_false))

    
    print ("[eps: %.3f][fool_rate: %.3f][true_adv: %d][false_adv: %d]"%(eps, fool_rate, cnt_true, cnt_false))
    

checkpoint["afs_inputs"] = torch.stack(afs_inputs)
checkpoint["afs_targets"] = torch.stack(afs_targets)
print (checkpoint["afs_inputs"].size(), checkpoint["afs_targets"].size())

if len(args.load_path2) > 0:
    checkpoint2 = torch.load(args.load_path2)
    net2 = checkpoint2['net']
    net2.afs_inputs = checkpoint["afs_inputs"] 
    net2.afs_targets = checkpoint["afs_targets"]

    test_afs(net2, logfile)

torch.save(checkpoint, os.path.join(args.save_dir, str(args.runname) + '.afs_nowm.t7'))
