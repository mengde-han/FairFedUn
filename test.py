#!/usr/bin/env python
# coding: utf-8

# Imports and Setup
from __future__ import print_function
from __future__ import print_function
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib

# from utils.train_util import train

matplotlib.use('Agg')
from membership_functions_mnist import *
import argparse

# use the static variables
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=3, sci_mode=True)
torch.backends.cudnn.enabled = True

class LeftDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        left_half = image[:, :, :image.size(2)//2]
        return left_half, label

class RightDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        right_half = image[:, :, image.size(2)//2:]
        return right_half, label
    
def train(net, epochs, lr, train_loader, test_loader, device='cuda', label_smoothing=0, warmup_step=0, warm_lr=10e-5):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """

    st = time.time()

    # print('==> Preparing data..')
    criterion = LabelSmoothCELoss(label_smoothing)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    warmup_scheduler = linear_warmup_scheduler(optimizer, warmup_step, warm_lr, lr)

    # best_acc = 0  # best test accuracy
    for epoch in range(0, epochs):
        """
        Start the training code.
        """
        # print('\nEpoch: %d' % epoch, '/ %d;' % epochs, 'learning_rate:', optimizer.state_dict()['param_groups'][0]['lr'])

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            net.train()
            # if warmup scheduler==None or not in scope of warmup -> if_warmup=False
            if_warmup = False if warmup_scheduler == None else warmup_scheduler.if_in_warmup()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if if_warmup:
                warmup_scheduler.step()

        torch.save(net.state_dict(), "ckpt/" + str(epoch)+"_original.plt")
        if not warmup_scheduler or not warmup_scheduler.if_in_warmup():
            scheduler.step()
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

class base_warmup():
    def __init__(self, optimizer, warm_step, warm_lr, dst_lr):
        ''' base class for warmup scheduler
        Args:
            optimizer: adjusted optimizer
            warm_step: total number of warm_step,(batch num)
            warm_lr: start learning rate of warmup
            dst_lr: init learning rate of train stage eg. 0.01
        '''
        assert warm_lr < dst_lr, "warmup lr must smaller than init lr"
        self.optimizer = optimizer
        self.warm_lr = warm_lr
        self.init_lr = dst_lr
        self.warm_step = warm_step
        self.stepped = 0
        if self.warm_step:
            self.optimizer.param_groups[0]['lr'] = self.warm_lr

    def step(self):
        self.stepped += 1

    def if_in_warmup(self) -> bool:
        return True if self.stepped < self.warm_step else False

    def set2initial(self):
        ''' Reset the learning rate to initial lr of training stage '''
        self.optimizer.param_groups[0]['lr'] = self.init_lr

    @property
    def now_lr(self):
        return self.optimizer.param_groups[0]['lr']


class linear_warmup_scheduler(base_warmup):
    def __init__(self, optimizer, warm_step, warm_lr, dst_lr):
        super().__init__(optimizer, warm_step, warm_lr, dst_lr)
        if (self.warm_step <= 0):
            self.inc = 0
        else:
            self.inc = (self.init_lr - self.warm_lr) / self.warm_step

    def step(self) -> bool:
        if (not self.stepped < self.warm_step): return False
        self.optimizer.param_groups[0]['lr'] += self.inc
        super().step()
        return True

    def still_in_warmup(self) -> bool:
        return True if self.stepped < self.warm_step else False


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        print('label smoothing:', self.smoothing)

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
                                         1.0 - self.smoothing) * one_hot_label + self.smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss
class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input size: [28, 14]
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.dropout = nn.Dropout2d(0.5)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool2d(x, 2)  # [14, 7]
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool2d(x, 2)  # [7, 3]
    #     x = F.relu(self.conv3(x))
    #     x = self.dropout(x)
    #     return x

class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()
        # self.fc = nn.Linear(128 * 7 * 3 * 2, 10) # times 2 because we are combining outputs from Party A and B
        # self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2688, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x_a, x_b):
        # Flatten the outputs from Party A and Party B
        x_a = x_a.view(x_a.size(0), -1)
        x_b = x_b.view(x_b.size(0), -1)
        
        # Concatenate the outputs
        x = torch.cat([x_a, x_b], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ClientA(nn.Module):
    def __init__(self):
        super(ClientA, self).__init__()
        self.partA = FirstNet()

    def forward(self, x):
        return self.partA(x)

class ClientB(nn.Module):
    def __init__(self):
        super(ClientB, self).__init__()
        self.partB = FirstNet()

    def forward(self, x):
        return self.partB(x)

class Server(nn.Module):
    def __init__(self):
        super(Server, self).__init__()
        self.partC = SecondNet()

    def forward(self, x_a, x_b):
        return self.partC(x_a, x_b)

class SplitNN(nn.Module):
    def __init__(self, clientA, clientB, server, clientA_optimizer, clientB_optimizer, server_optimizer):
        super(SplitNN, self).__init__()
        self.clientA = clientA.to(args.device )
        self.clientB = clientB.to(args.device )
        self.server = server.to(args.device)
        self.clientA_optimizer = clientA_optimizer
        self.clientB_optimizer = clientB_optimizer
        self.server_optimizer = server_optimizer

    def forward(self, clientA_input, clientB_input):
        outputA = self.clientA(clientA_input)
        outputB = self.clientB(clientB_input)
        return self.server(outputA, outputB)

    # def backward(self, loss):
    #     server_grad = torch.autograd.grad(loss, self.server.partC.parameters(), retain_graph=True)
    #     clientA_grad = torch.autograd.grad(server_grad, self.clientA.partA.parameters(), retain_graph=True)
    #     clientB_grad = torch.autograd.grad(server_grad, self.clientB.partB.parameters(), retain_graph=True)
    #     return server_grad, clientA_grad, clientB_grad

    def step(self):
        self.clientA_optimizer.step()
        self.clientB_optimizer.step()
        self.server_optimizer.step()

    def zero_grads(self):
        self.clientA_optimizer.zero_grad()
        self.clientB_optimizer.zero_grad()
        self.server_optimizer.zero_grad()

def generate(dataset, list_classes: list):
    sub_dataset = []
    for datapoint in dataset:
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            sub_dataset.append(datapoint)

    '''
    for i in range(0, len(dataset)):
        if in_data[i][1] in list_classes:
            sub_dataset.append(in_data[i])
    '''
    return sub_dataset


def args_parser():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--batch_size_train', type=int, default=64, help='model to use')
    parser.add_argument('--total_class', type=int, default=10, help='model to use')
    parser.add_argument('--unlearning_class', type=int, default=8, help='model to use')

    # Training Shadow Models
    parser.add_argument('--num_shadow_models', type=int, default=10, help='model to use')
    parser.add_argument('--shadow_training_epochs', type=int, default=10, help='model to use')
    parser.add_argument('--shadow_model_training_batchsize', type=int, default=64, help='model to use')

    # Training Attack Model
    parser.add_argument('--attack_model_training_batchsize', type=int, default=64, help='model to use')

    parser.add_argument('--dataset', type=str, default='mnist', help='cifar10 or imagenet')
    parser.add_argument('--dataroot', type=str, default="../data/MNIST",
                        metavar='PATH',
                        help='Path to Dataset folder')
    parser.add_argument('--model', type=str, default='resnet20', help='model to use')

    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--bs', type=int, default=256, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # my unlearning
    parser.add_argument('--label_smoothing', type=float, default=0.0, 
                        help='label smoothing rate')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='warm up epochs')
    parser.add_argument('--warm_lr', type=float, default=10e-5,
                        help='warm up learning rate')

    args = parser.parse_args()
    return args


def trainwithattack(args, net, epochs, lr, train_loader, test_loader, device='cuda', label_smoothing=0, warmup_step=0, warm_lr=10e-5, atk_loader=None, attack_model=None):
    # print('==> Preparing data..')
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    warmup_scheduler = linear_warmup_scheduler(optimizer, warmup_step, warm_lr, lr)

    # best_acc = 0  # best test accuracy
    for epoch in range(0, 2):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            net.cuda()
            net.train()
            # if warmup scheduler==None or not in scope of warmup -> if_warmup=False
            if_warmup = False if warmup_scheduler == None else warmup_scheduler.if_in_warmup()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if if_warmup:
                warmup_scheduler.step()
            if batch_idx % 10 == 0:
                print(batch_idx, testtargetmodel_10(args, net, atk_loader, attack_model))
        if not warmup_scheduler or not warmup_scheduler.if_in_warmup():
            scheduler.step()

# def imshow(img):
#     # 如果使用了归一化，需要反转这个过程
#     img = img / 2 + 0.5     # 反归一化
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

if __name__ == '__main__':

    args = args_parser()
    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.random.manual_seed(42)
    # Data Entry and Processing
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST(args.dataroot, train=True, download=True, transform=trans_mnist)
        testset = datasets.MNIST(args.dataroot, train=False, download=True, transform=trans_mnist)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    # Split the training dataset to "in" dataset and "out" dataset
    in_data, out_data = torch.utils.data.random_split(trainset, [int(len(trainset) / 2), int(len(trainset) / 2)])

    # Split the in_data dataset to "in" dataset and "out" dataset
    target_index = []
    nontarget_index = []
    for i in range(0, len(in_data)):
        # print(f"Index: {i}, Label: {in_data[i][1]}")
        if in_data[i][1] == args.unlearning_class:
            target_index.append(i)
        else:
            nontarget_index.append(i)
    # 创建针对target_index的Subset
    target_subset = Subset(in_data, target_index)
    nontarget_subset = Subset(in_data, nontarget_index)

    # 创建左右数据集
    target_left_dataset = LeftDataset(target_subset)
    target_right_dataset = RightDataset(target_subset)
    nontarget_left_dataset = LeftDataset(nontarget_subset)
    nontarget_right_dataset = RightDataset(nontarget_subset)
    # image, label = left_dataset[2]

    # # 显示图像的标签
    # print("Label of the image:", label)
    # image = image.squeeze(0)  # This reduces the (1, 28, 14) tensor to (28, 14)

    # # 显示图像
    # plt.imshow(image, cmap='gray')  # 确保以灰度模式显示
    # plt.savefig("left.png")
    # # 创建DataLoader
    target_left_loader = DataLoader(target_left_dataset, batch_size=64)
    target_right_loader = DataLoader(target_right_dataset, batch_size=64)

    nontarget_left_loader = DataLoader(nontarget_left_dataset, batch_size=64)
    nontarget_right_loader = DataLoader(nontarget_right_dataset, batch_size=64)

    # Split the testdata dataset to "in" dataset and "out" dataset
    eights_index = []
    noneights_index = []
    for i in range(0, len(testset)):
        if testset[i][1] == args.unlearning_class:
            eights_index.append(i)
        else:
            noneights_index.append(i)
    
    eights_subset = Subset(in_data, eights_index)
    noneights_subset = Subset(in_data, noneights_index)

    # 创建左右数据集
    eights_left_dataset = LeftDataset(eights_subset)
    eights_right_dataset = RightDataset(eights_subset)
    noneights_left_dataset = LeftDataset(noneights_subset)
    noneights_right_dataset = RightDataset(noneights_subset)
    # image, label = left_dataset[2]

    # # 显示图像的标签
    # print("Label of the image:", label)
    # image = image.squeeze(0)  # This reduces the (1, 28, 14) tensor to (28, 14)

    # # 显示图像
    # plt.imshow(image, cmap='gray')  # 确保以灰度模式显示
    # plt.savefig("left.png")
    # # 创建DataLoader
    eights_left_loader = DataLoader(eights_left_dataset, batch_size=64)
    eights_right_loader = DataLoader(eights_right_dataset, batch_size=64)

    noneights_left_loader = DataLoader(noneights_left_dataset, batch_size=64)
    noneights_right_loader = DataLoader(noneights_right_dataset, batch_size=64)

    # Create shadow datasets. Each must have an "in" and "out" set for attack model dataset generation ([in, out]). Each shadow model is trained only on the "in" data.
    shadow_datasets = []

    for i in range(args.num_shadow_models):
        shadow_datasets.append(
            torch.utils.data.random_split(trainset, [int(len(trainset) / 2), int(len(trainset) / 2)]))

    # train and save shadowmodels
    shadow_models = []
    for _ in range(args.num_shadow_models):
        shadow_models.append(target_model_fn(args))

    for i, shadow_model_set in enumerate(shadow_models):
        shadow_model = shadow_model_set[0]
        shadow_optim = shadow_model_set[1]
        shadow_model_in_loader = torch.utils.data.DataLoader(shadow_datasets[i][0],
                                                             batch_size=args.shadow_model_training_batchsize,
                                                             shuffle=True)
        print(f"Training shadow model {i}")
        for epoch in range(1, args.shadow_training_epochs + 1):
            print(f"\r\tEpoch {epoch}  ", end="")
            shadow_train(shadow_model, shadow_optim, epoch, shadow_model_in_loader)
            if epoch == args.shadow_training_epochs:
                test(shadow_model, testloader, dname="All data")
        path = F"infattack_10/resnet-shadow_model_{i}.pt"
        torch.save({'model_state_dict': shadow_model.state_dict(), }, path)

    # load shadowmodels
    shadow_models = []
    for i in range(args.num_shadow_models):
        shadow_model = target_model_fn(args)
        path = F"infattack_10/resnet-shadow_model_{i}.pt"
        checkpoint = torch.load(path)
        shadow_model[0].load_state_dict(checkpoint['model_state_dict'])
        shadow_models.append(shadow_model)

    # Generate attack training set for current class
    sm = nn.Softmax()
    attack_x = []
    attack_y = []
    for i, shadow_model_set in enumerate(shadow_models):
        print(f"\rGenerating class {args.unlearning_class} set from model {i}", end="")
        shadow_model = shadow_model_set[0].eval().cpu()
        in_loader = torch.utils.data.DataLoader(shadow_datasets[i][0], batch_size=1)
        for data, target in in_loader:
            if target == args.unlearning_class:
                pred = shadow_model(data).view(args.total_class)
                if torch.argmax(pred).item() == args.unlearning_class:
                    attack_x.append(sm(pred))
                    attack_y.append(1)
        out_loader = torch.utils.data.DataLoader(shadow_datasets[i][1], batch_size=1)
        for data, target in out_loader:
            if target == args.unlearning_class:
                pred = shadow_model(data).view(args.total_class)
                attack_x.append(sm(pred))
                attack_y.append(0)
    # Save datasets
    tensor_x = torch.stack(attack_x)
    tensor_y = torch.Tensor(attack_y)
    xpath = f"infattack_10/resnet_attack_x_{args.unlearning_class}.pt"
    ypath = f"infattack_10/resnet_attack_y_{args.unlearning_class}.pt"
    torch.save(tensor_x, xpath)
    torch.save(tensor_y, ypath)

    # Load datasets
    tensor_x = torch.load(f"infattack_10/resnet_attack_x_{args.unlearning_class}.pt")
    tensor_y = torch.load(f"infattack_10/resnet_attack_y_{args.unlearning_class}.pt")

    # Create test and train dataloaders for attack dataset
    attack_datasets = []
    attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
    attack_train, attack_test = torch.utils.data.random_split(attack_datasets[0], [int(0.9 * len(attack_datasets[0])),
                                                                                   len(attack_datasets[0]) - int(
                                                                                       0.9 * len(attack_datasets[0]))])
    attackloader = torch.utils.data.DataLoader(attack_train, batch_size=args.attack_model_training_batchsize,
                                               shuffle=True)
    attacktester = torch.utils.data.DataLoader(attack_test, batch_size=args.attack_model_training_batchsize,
                                               shuffle=True)
    # Create and train an attack model
    attack_model, attack_optimizer = attack_model_fn_10()
    for epoch in range(10):
        trainattacker(attack_model, attack_optimizer, epoch, attackloader)
    print(fulltestattacker(attack_model, attacktester))
    # Save attack model
    path = F"infattack_10/resnet_attack_model_{args.unlearning_class}.pt"
    torch.save({'model_state_dict': attack_model.state_dict(), }, path)

    # read
    attack_model, attack_optimizer = attack_model_fn_10()
    path = F"infattack_10/resnet_attack_model_{args.unlearning_class}.pt"
    checkpoint = torch.load(path)
    attack_model.load_state_dict(checkpoint['model_state_dict'])

##########################################
    fusion_types = ['FedAvg','Retrain']
    clientA = ClientA().to(args.device)
    clientB = ClientB().to(args.device)
    server = Server().to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizerA = torch.optim.Adam(clientA.parameters(), lr=0.001)
    optimizerB = torch.optim.Adam(clientB.parameters(), lr=0.001)
    optimizerServer = torch.optim.Adam(server.parameters(), lr=0.001)
    initial_model = SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(args.device)
    model_dict = {}
    for fusion_key in fusion_types:
    
        model_structure = copy.deepcopy(initial_model)  # 修改为实际模型结构
        model_structure.load_state_dict(torch.load(f'{fusion_key}_weights.pth',map_location=args.device))
        model_dict[fusion_key] = model_structure.state_dict()


    # loaded_state_dicts_dict = torch.load('party_models_dict.pth',map_location=args.device )
    # party_models_dict = {}

    # for key, state_dicts_list in loaded_state_dicts_dict.items():
    #     loaded_models_list = []
    #     for state_dict in state_dicts_list:
    #         model_instance = copy.deepcopy(initial_model)   # 请替换为你的实际模型结构
    #         model_instance.load_state_dict(state_dict)
    #         loaded_models_list.append(model_instance)
        
    #     party_models_dict[key] = loaded_models_list

    fedavg_model_state_dict = copy.deepcopy(model_dict['FedAvg'])
    fedavg_model = copy.deepcopy(initial_model)
    fedavg_model.load_state_dict(fedavg_model_state_dict)

    retrain_model_state_dict = copy.deepcopy(model_dict['Retrain'])
    retrain_model = copy.deepcopy(initial_model)
    retrain_model.load_state_dict(fedavg_model_state_dict)
######################################
    # if args.model == "resnet20":
    #     net_glob_original = models.resnet18(weights=None).cuda()
    #     net_glob_original.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #     net_glob_original.fc = nn.Sequential(nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    #     net_glob_original = net_glob_original.cuda()
    # else:
    #     print('no model')

    # in_loader = torch.utils.data.DataLoader(in_data, batch_size=args.batch_size_train, shuffle=True)

    # train(net_glob_original, epochs=20,
    #       lr=args.lr,
    #       train_loader=in_loader,
    #       test_loader=testloader,
    #       label_smoothing=args.label_smoothing,
    #       warmup_step=args.warmup_step,
    #       warm_lr=args.warm_lr)

    # # save TargetModel
    # path = F"infattack_10/net_glob_original.pt"
    # torch.save({'model_state_dict': net_glob_original.state_dict()}, path)

    # net_glob_original = models.resnet18(weights=None).cuda()
    # net_glob_original.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # net_glob_original.fc = nn.Sequential(nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    # net_glob_original = net_glob_original.cuda()

    # path = F"infattack_10/net_glob_original.pt"
    # checkpoint = torch.load(path)
    # net_glob_original.load_state_dict(checkpoint['model_state_dict'])

    final_in_loader = torch.utils.data.DataLoader(in_data, batch_size=1, shuffle=False)
    final_out_loader = torch.utils.data.DataLoader(out_data, batch_size=1, shuffle=False)
    attack_datasets = []
    sm = nn.Softmax()
    for c in range(args.unlearning_class, args.unlearning_class + 1):
        fedavg_model.eval().cpu()
        attackdata_x = []
        attackdata_y = []
        count = 0
        print(f"\rGenerating class {args.unlearning_class} set from target model", end="")
        for data, target in final_in_loader:
            if target == args.unlearning_class:
                pred = fedavg_model(data).view(args.total_class)
                if torch.argmax(pred).item() == args.unlearning_class:
                    attackdata_x.append(data)
                    attackdata_y.append(1)
                    count += 1
        for data, target in final_out_loader:
            if target == c:
                attackdata_x.append(data)
                attackdata_y.append(0)
                count += 1
        attack_tensor_x = torch.stack(attackdata_x)
        attack_tensor_y = torch.Tensor(attackdata_y)

    atk_data = torch.utils.data.TensorDataset(attack_tensor_x, attack_tensor_y)
    atk_loader = torch.utils.data.DataLoader(atk_data, batch_size=1, shuffle=False)
    print(testtargetmodel_10(args, fedavg_model, atk_loader, attack_model))



# ###########################################################
#     # reTrain TargetModel
#     cprint("Finetuning")
#     list_allclasses = list(range(args.total_class))
#     list_allclasses.remove(args.unlearning_class)
#     unlearn_listclass = [args.unlearning_class]
#     rest_testdata = generate(in_data, list_allclasses)
#     in_loader = torch.utils.data.DataLoader(rest_testdata, batch_size=args.batch_size_train, shuffle=True)

#     if args.model == "resnet20":
#         net_glob_finetuning = models.resnet18(weights=None).cuda()
#         net_glob_finetuning.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         net_glob_finetuning.fc = nn.Sequential(nn.Linear(512, 10), nn.LogSoftmax(dim=1))
#         net_glob_finetuning = net_glob_finetuning.cuda()

#         path = F"infattack_10/net_glob_original.pt"
#         checkpoint = torch.load(path)
#         net_glob_finetuning.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         print('no model')

#     trainwithattack(args, net_glob_finetuning, epochs=1,
#                     lr=args.lr,
#                     train_loader=in_loader,
#                     test_loader=testloader,
#                     label_smoothing=args.label_smoothing,
#                     warmup_step=args.warmup_step,
#                     warm_lr=args.warm_lr,
#                     atk_loader=atk_loader,
#                     attack_model=attack_model
#                     )

#     cprint("reTrain")
#     list_allclasses = list(range(args.total_class))
#     list_allclasses.remove(args.unlearning_class)
#     unlearn_listclass = [args.unlearning_class]
#     rest_testdata = generate(in_data, list_allclasses)
#     in_loader = torch.utils.data.DataLoader(rest_testdata, batch_size=args.batch_size_train, shuffle=True)

#     if args.model == "resnet20":
#         net_glob_retraining = models.resnet18(weights=None).cuda()
#         net_glob_retraining.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         net_glob_retraining.fc = nn.Sequential(nn.Linear(512, 10), nn.LogSoftmax(dim=1))
#         net_glob_retraining = net_glob_retraining.cuda()

#     else:
#         print('no model')

#     trainwithattack(args, net_glob_retraining, epochs=1,
#                     lr=args.lr,
#                     train_loader=in_loader,
#                     test_loader=testloader,
#                     label_smoothing=args.label_smoothing,
#                     warmup_step=args.warmup_step,
#                     warm_lr=args.warm_lr, atk_loader=atk_loader,
#                     attack_model=attack_model)

    # unlearning

    '''
    True Positive（TP）：真正类。样本的真实类别是正类，并且模型识别的结果也是正类。
    False Negative（FN）：假负类。样本的真实类别是正类，但是模型将其识别为负类。
    False Positive（FP）：假正类。样本的真实类别是负类，但是模型将其识别为正类。
    True Negative（TN）：真负类。样本的真实类别是负类，并且模型将其识别为负类。 
    '''

    '''
    Precision：推断是训练数据的数据占实际的
    （what fraction of records inferred as members are indeed members of the training dataset）
    Recall：训练集中的数据有多少被判断为成员
    （what fraction of the training dataset’s members are correctly inferred as members by the attackers）
    '''