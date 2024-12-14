#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from __future__ import print_function
import math
from collections import OrderedDict

import copy
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from cprint import cprint
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms, models
# from utils.resnet_cifar import ResNet_CIFAR

torch.set_printoptions(precision=3, sci_mode=True)
cuda = True if torch.cuda.is_available() else False


def normalize(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    trans = np.transpose(npimg, (1, 2, 0))
    return np.squeeze(trans)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# FCNN attack model for membership inference attack
class AttackModel_100(nn.Module):
    def __init__(self):
        super(AttackModel_100, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)


class AttackModel_10(nn.Module):
    def __init__(self):
        super(AttackModel_10, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)

def attack_model_fn_10():
    model = AttackModel_10()
    optimizer = optim.Adam(model.parameters())
    return model, optimizer

# Function to generate attack models
def attack_model_fn_100():
    model = AttackModel_100()
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


# target models
def target_model_fn(args):
    '''
    # load resnet 18 and change to fit problem dimensionality
    resnet = models.resnet18()
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.fc = nn.Sequential(nn.Linear(512, args.num_classes))
    optimizer = optim.Adam(resnet.parameters())
    '''
    if args.model == 'resnet20' and args.dataset =="cifar10":
        net_glob = ResNet_CIFAR(depth=20, num_classes=10).cuda()
        optimizer = optim.Adam(net_glob.parameters())
    if args.model == 'resnet20' and args.dataset =="cifar100":
        net_glob = ResNet_CIFAR(depth=20, num_classes=100).cuda()
        optimizer = optim.Adam(net_glob.parameters())
    if args.model == 'resnet20' and args.dataset =="mnist":
        net_glob = AlexNetCIFAR()
        optimizer = optim.Adam(net_glob.parameters())
    return net_glob, optimizer
  
class FLNet(nn.Module):
    def __init__(self):
        super(FLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCIFAR, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x
    
# Training method for attack model
def trainattacker(model, optimizer, epoch, loader):
    device = "cuda"
    model.train().cuda()
    batches = []
    steps = []
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.flatten(output)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(
            "\rEpoch: {} [{:6d}]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), loss.item() / len(loader.dataset)),
            end="")
    return batches, steps


# Testing method for attack model
def testattacker(model, loader, dname="Test set"):
    device = "cuda"
    model.eval().cuda()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(data)
            output = torch.flatten(output)
            total += target.size()[0]
            test_loss += F.binary_cross_entropy(output, target).item()
            pred = torch.round(output)
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)

    print('{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        dname, test_loss, correct, total,
        100. * correct / total
    ))
    return 1. * correct / total


# Testing method for attack that returns full confusion matrix
def fulltestattacker(model, loader):
    device = "cuda"
    model.eval().cuda()
    with torch.no_grad():
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.flatten(output)
            pred = torch.round(output)
            #       correct += pred.eq(target.data.view_as(pred)).sum()
            for i in range(len(pred)):
                if pred[i] == target[i] == 1:
                    tp += 1
                if pred[i] == target[i] == 0:
                    tn += 1
                if pred[i] == 1 and target[i] == 0:
                    fp += 1
                if pred[i] == 0 and target[i] == 1:
                    fn += 1
    return tp, tn, fp, fn


# Training method
def shadow_train(model, optimizer, epoch, loader):
    device = "cuda"
    criterion = nn.CrossEntropyLoss()
    model.train().cuda()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch: {} [{:6d}]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), loss.item()))


# Testing method
def test(model, loader, dname="Test set"):
    device = "cuda"
    criterion = nn.CrossEntropyLoss()
    model.eval().cuda()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += target.size()[0]
            test_loss += criterion(output, target).item()
            _, pred = torch.topk(output, 10, dim=1, largest=True, sorted=True)
            for i, t in enumerate(target):
                if t in pred[i]:
                    correct += 1
    test_loss /= len(loader.dataset)
    print('{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(dname, test_loss, correct, total,
                                                                    100. * correct / total))
    return 1. * correct / total


# execute attack
def testtargetmodel_100(args, targetmodel, atk_loader, attack_model):
    sm = nn.Softmax()
    for c in range(args.unlearning_class, args.unlearning_class + 1):
        targetmodel.eval().cpu()
        attack_x = []
        attack_y = []
        for data, target in atk_loader:
            data = data.reshape(1, 3, 32, 32)
            pred = targetmodel(data).view(100)
            attack_x.append(sm(pred))
            attack_y.append(target)
        tensor_x = torch.stack(attack_x)
        tensor_y = torch.Tensor(attack_y)

        path = F"infattack_100/resnet_attack_model_{c}.pt"
        checkpoint = torch.load(path)
        attack_model.load_state_dict(checkpoint['model_state_dict'])

        attack_datasets = []
        attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
        attacktester = torch.utils.data.DataLoader(attack_datasets[0], batch_size=128, shuffle=True)

        tp, tn, fp, fn = fulltestattacker(attack_model, attacktester)
        recall = tp / (tp + fn)
        missrate = fn / (fn + tp)
        return recall

def testtargetmodel_10(args, targetmodel, atk_loader, attack_model):
    sm = nn.Softmax(dim=0)
    for c in range(args.unlearning_class, args.unlearning_class + 1):
        targetmodel.eval().cpu()
        attack_x = []
        attack_y = []
        for data, target in atk_loader:
            # data = data.reshape(1, 1, 28, 28)
            data = data.reshape(1, 3, 32, 32)
            pred = targetmodel(data).view(10)
            attack_x.append(sm(pred))
            attack_y.append(target)
        tensor_x = torch.stack(attack_x)
        tensor_y = torch.Tensor(attack_y)

        path = F"infattack_10/resnet_attack_model_{c}.pt"
        checkpoint = torch.load(path)
        attack_model.load_state_dict(checkpoint['model_state_dict'])

        attack_datasets = []
        attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
        attacktester = torch.utils.data.DataLoader(attack_datasets[0], batch_size=128, shuffle=True)

        tp, tn, fp, fn = fulltestattacker(attack_model, attacktester)
        print(tp,tn, fp,fn)
        recall = tp / (tp + fn)
        #print(f"\trecall: {recall}")
        missrate = fn / (fn + tp)
        #print(f"\tmissrate: {missrate}")
        
        return recall

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def cutout_func(img, length=16):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    # mask = torch.from_numpy(mask)
    mask = mask.reshape(img.shape)
    img *= mask
    return img


def cutout_batch(img, length=16):
    h, w = img.size(2), img.size(3)
    masks = []
    for i in range(img.size(0)):
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img[0]).unsqueeze(0)
        masks.append(mask)
    masks = torch.cat(masks).cuda()
    img *= masks
    return img


class CIFAR_INPUT_ITER():
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, batch_size, data_type='train', root='/userhome/data/cifar10'):
        self.root = root
        self.batch_size = batch_size
        self.train = (data_type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        np.save("cifar.npy", self.data)
        self.data = np.load('cifar.npy')  # to serialize, increase locality

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.data, self.targets = shuffle(
                    self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__


class MyNetwork(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def feature_extract(self, x):
        raise NotImplementedError

    @property
    def config(self):  # should include name/cfg/cfg_base/dataset
        raise NotImplementedError

    def cfg2params(self, cfg):
        raise NotImplementedError

    def cfg2flops(self, cfg):
        raise NotImplementedError

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def init_model(self, model_init, init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data)
                elif model_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data)
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_parameters(self, keys=None, mode='include'):
        if keys is None:
            for name, param in self.named_parameters():
                yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def weight_parameters(self):
        return self.get_parameters()

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes)
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes)

        self.conv_bn1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.conv_bn2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-in_planes)//2, planes-in_planes-(planes-in_planes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-in_planes)//2, planes-in_planes-(planes-in_planes)//2), "constant", 0))

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(MyNetwork):
    def __init__(self, depth=20, num_classes=10, cfg=None, cutout=True):
        super(ResNet_CIFAR, self).__init__()
        if cfg is None:
            cfg = [16, 16, 32, 64]
        num_blocks = []
        if depth==20:
            num_blocks = [3, 3, 3]
        elif depth==32:
            num_blocks = [5, 5, 5]
        elif depth==44:
            num_blocks = [7, 7, 7]
        elif depth==56:
            num_blocks = [9, 9, 9]
        elif depth==110:
            num_blocks = [18, 18, 18]
        block = BasicBlock
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cutout = cutout
        self.cfg = cfg
        self.in_planes = 16
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        self.conv_bn = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.layer1 = self._make_layer(block, cfg[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, cfg[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[3], num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            layers.append(('block_%d'%i, block(self.in_planes, planes, strides[i])))
            self.in_planes = planes
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        out = F.relu(self.conv_bn(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def cfg2params(self, cfg):
        params = 0
        params += (3 * 3 * 3 * cfg[0] + cfg[0] * 2) # conv1+bn1
        in_c = cfg[0]
        cfg_idx = 1
        for i in range(3):
            num_blocks = self.num_blocks[i]
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                params += (in_c * 3 * 3 * c + 2 * c + c * 3 * 3 * c + 2 * c) # per block params
                if in_c != c:
                    params += in_c * c # shortcut
                in_c = c
            cfg_idx += 1
        params += (self.cfg[-1] + 1) * self.num_classes # fc layer
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        size = 32
        flops = 0
        flops += (3 * 3 * 3 * cfg[0] * 32 * 32 + cfg[0] * 32 * 32 * 4) # conv1+bn1
        in_c = cfg[0]
        cfg_idx = 1
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                flops += (in_c * 3 * 3 * c * size * size + c * size * size * 4 + c * 3 * 3 * c * size * size + c * size * size * 4) # per block flops
                if in_c != c:
                    flops += in_c * c * size * size # shortcut
                in_c = c
            cfg_idx += 1
        flops += (2 * self.cfg[-1] + 1) * self.num_classes # fc layer
        return flops

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': [16, 16, 32, 64],
            'dataset': 'cifar10',
        }

def ResNet20():
    return ResNet_CIFAR(depth=20)
def ResNet32():
    return ResNet_CIFAR(depth=32)
def ResNet44():
    return ResNet_CIFAR(depth=44)
def ResNet56():
    return ResNet_CIFAR(depth=56)
def ResNet110():
    return ResNet_CIFAR(depth=110)
