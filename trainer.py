import numpy as np
import torch
from torch.autograd import Variable

from helpers.utils import progress_bar
from helpers.loaders import batch_gen

# Train function

def CrossEnt(x, y):
    return (- x * torch.log(y.clamp(min=1e-7))).sum()

def IsInside(x, Y):
    for y in Y:
        if x is y:
            return True
    return False

def RandomTransform(x, device):
    x = x + torch.cuda.FloatTensor(x.size()).normal_(0, 0.05)
    theta = torch.zeros((x.size(0), 2, 3)).to(device)
    sign = (torch.randint(0, 2, size=(x.size(0), 1, 1), dtype=torch.float) * 2 - 1).to(device)
    theta[:, 0:1, 0:1] = torch.cuda.FloatTensor(x.size(0), 1, 1).normal_(1, 0.1) * sign
    theta[:, 1:2, 1:2] = torch.cuda.FloatTensor(x.size(0), 1, 1).normal_(1, 0.1)
    
    
    return torch.nn.functional.grid_sample(x, grid = torch.nn.functional.affine_grid(theta, x.size()))
    

def train(epoch, net, criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True, ex_datas = [], ex_net = None, wm2_loader = None, n_classes=None, EWC_coef = 0., Fisher = None, init_params = None, EWC_immune = [], afs_bsize=0, extra_only = False):
    print('\nEpoch: %d' % epoch)
    
    net.train()
    train_loss = 0
    train_loss_wm = 0
    correct = 0
    total = 0
    iteration = -1
    wm_correct = 0
    print_every = 5
    l_lambda = 1.2

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images

    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    if afs_bsize > 0:
        afs_idx = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        if afs_bsize > 0:
            inputs = torch.cat([inputs, net.afs_inputs[afs_idx:afs_idx + afs_bsize]], dim = 0)
            targets = torch.cat([targets, net.afs_targets[afs_idx:afs_idx + afs_bsize]], dim=0)
            afs_idx = (afs_idx + afs_bsize) % net.afs_inputs.size(0)


        # add data from extra sources
        original_batch_size = targets.size(0)
        extra_only_tag = True
        for _loader in ex_datas:
            _input, _target = next(_loader)
            _input, _target = _input.to(device), _target.to(device)
            if _target[0].item() < -1:
                with torch.no_grad():
                    _, __target = torch.max(ex_net(_input).data, 1)
                    _target = (__target + _target + 20000)%n_classes
            elif _target[0].item() == -1 or ex_net!=None:
                with torch.no_grad():
                    _output = ex_net(_input)
                    
                    _, _target = torch.max(_output.data, 1)
                    _target = _target.to(device)

            if extra_only and extra_only_tag:
                inputs = _input
                targets = _target
                extra_only_tag = False
            else:
                inputs = torch.cat([inputs, _input], dim=0)
                targets = torch.cat([targets, _target], dim=0)


        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if EWC_coef > 0:
            for param, fisher, init_param in zip(net.parameters(), Fisher, init_params):
                if IsInside(param, EWC_immune):
                    continue
                loss = loss + (0.5 * EWC_coef * fisher.clamp(max = 1. / optimizer.param_groups[0]['lr'] / EWC_coef) * ((param - init_param)**2)).sum()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
                
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))


# train function in a teacher-student fashion
def train_teacher(epoch, net, criterion, optimizer, use_cuda, logfile, loader, wmloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            if use_cuda:
                wminput, wmtarget = wminput.cuda(), wmtarget.cuda()
            wminputs.append(wminput)
            wmtargets.append(wmtarget)
        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if wmloader:
            # add wmimages and targets
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

def test_afs(net, logfile):
    net.eval()
    inputs, targets = net.afs_inputs, net.afs_targets
    criterion = torch.nn.CrossEntropyLoss()    
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        correct = predicted.eq(targets.data).cpu().sum()
        total = inputs.size(0)
    with open(logfile, 'a') as f:
        f.write('Test(afw) results:\n')
        print('Test(afw) results:')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (loss, 100. * float(correct) / total, correct, total))
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss, 100. * float(correct) / total, correct, total))

# Test function
def test(net, criterion, logfile, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, targets)
        correct += predicted.eq(targets.data).cpu().sum()

        test_loss += loss.item()
        total += targets.size(0)        
        
        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
    # return the acc.
    return 100. * correct / total
