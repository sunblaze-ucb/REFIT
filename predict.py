from __future__ import print_function

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from helpers.loaders import *
from helpers.utils import progress_bar

parser = argparse.ArgumentParser(description='Test models on testing sets and watermark sets.')
parser.add_argument('--model_path', default='checkpoint/teacher-cifar100-2.t7', help='the model path')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--testwm', action='store_true', help='test the wm set or the testing set.')
parser.add_argument('--db_path', default='./data', help='the path to the root folder of the test data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to use')
parser.add_argument('--label_mapping', action='store_true', help='mapping the dataset label from cifar-10 to stl10')
parser.add_argument('--predict_path', default='labels-pred.txt', help='the path to the predicted labels')

label_mp = [0, 2, 1, 3, 4, 5, -1, 6, 8, 9]
#stl-10:airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
#cifar-10:airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 100

# Data
if args.testwm:
    print('Loading watermark images')
    loader = getwmloader(args.wm_path, batch_size, args.wm_lbl, shuffle=False)
    _, _, num_classes = getdataloader(args.dataset, args.db_path, args.db_path, batch_size)
else:
    _, loader, num_classes = getdataloader(args.dataset, args.db_path, args.db_path, batch_size)

# Model
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.exists(args.model_path), 'Error: no checkpoint found!'
checkpoint = torch.load(args.model_path, map_location=device)
net = checkpoint['net']
acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

net.eval()
test_loss = 0
correct = 0
total = 0

correct_class = torch.zeros(num_classes)
total_class = torch.zeros(num_classes)

predicted_labels = []

for batch_idx, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
#    print (inputs.size())
    with torch.no_grad():
        outputs = net(inputs)
        loss = criterion(outputs, targets)

    test_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)

    predicted_labels += predicted.data.cpu().numpy().tolist()

    if args.label_mapping:
        targets = torch.LongTensor([label_mp[_] for _ in targets.data], )
        correct += predicted.cpu().eq(targets).sum()
    else:
        correct += predicted.eq(targets.data).cpu().sum()
            
    for label in range(num_classes):
        correct_class[label] += (predicted.eq(targets.data) * (targets.data.eq(label))).cpu().sum()
        total_class[label] += targets.data.eq(label).cpu().sum()

    progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

print (correct_class / total_class)
print (total_class)

with open(args.predict_path, 'w') as f:
    for i, pred in enumerate(predicted_labels):
        if i != 0:
            f.write('\n')
        f.write(str(pred))

