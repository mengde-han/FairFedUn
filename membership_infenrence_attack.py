#!/usr/bin/env python
# coding: utf-8

# Imports and Setup
from __future__ import print_function
from __future__ import print_function

import matplotlib
import abc
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

class LocalTraining():

    """
    Base class for Local Training
    """

    def __init__(self, 
                 num_updates_in_epoch=None,
                 num_local_epochs=1):
       
        self.name = "local-training"
        self.num_updates = num_updates_in_epoch
        self.num_local_epochs = num_local_epochs
        

    def train(self, model, trainloader, criterion=None, opt=None, lr = 1e-2):
        """
        Method for local training
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if opt is None:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        if self.num_updates is not None:
            self.num_local_epochs = 1
        device='cuda'
        model.cuda()
        model.train()
        running_loss = 0.0
        for epoch in range(self.num_local_epochs):
            for batch_idx, (data, target) in enumerate(trainloader):

                x_batch, y_batch = data.to(device), target.to(device)

                opt.zero_grad()

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                opt.step()
                
                running_loss += loss.item()

        return model
    
class Fusion(abc.ABC):

    """
    Base class for Fusion
    """

    def __init__(self, num_parties):
        self.name = "fusion"
        self.num_parties = num_parties
        
    def average_selected_models(self, selected_parties, party_models):
        with torch.no_grad():
            sum_vec = nn.utils.parameters_to_vector(party_models[selected_parties[0]].parameters())
            if len(selected_parties) > 1:
                for i in range(1,len(selected_parties)):
                    sum_vec += nn.utils.parameters_to_vector(party_models[selected_parties[i]].parameters())
                sum_vec /= len(selected_parties)

            model = copy.deepcopy(party_models[0])
            nn.utils.vector_to_parameters(sum_vec, model.parameters())
        return model.state_dict()
            
            
    @abc.abstractmethod
    def fusion_algo(self, party_models, current_model=None):
        raise NotImplementedError


class FusionAvg(Fusion):

    def __init__(self, num_parties):
        super().__init__(num_parties)
        self.name = "Fusion-Average"

    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(self.num_parties)]
        aggregated_model_state_dict = super().average_selected_models(selected_parties, party_models)
        return aggregated_model_state_dict 


class FusionRetrain(Fusion):

    def __init__(self, num_parties):
        super().__init__(num_parties)
        self.name = "Fusion-Retrain"
        
    # Currently, we assume that the party to be erased is party_id = 0
    def fusion_algo(self, party_models, current_model=None):
        selected_parties = [i for i in range(1,self.num_parties)]
        aggregated_model_state_dict = super().average_selected_models(selected_parties, party_models)
        return aggregated_model_state_dict 

def FL_round_fusion_selection(num_parties, fusion_key='FedAvg'):

    fusion_class_dict = {
        'FedAvg': FusionAvg(num_parties),
        'Retrain': FusionRetrain(num_parties), 
        'Unlearn': FusionAvg(num_parties)
        }

    return fusion_class_dict[fusion_key]

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

def get_distance(model1, model2):
        model1=model1.cuda()
        model2=model2.cuda()
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance.cpu()
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
    parser.add_argument('--shadow_training_epochs', type=int, default=1, help='model to use')
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


if __name__ == '__main__':
    torch.random.manual_seed(42)
    args = args_parser()
    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

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
        if in_data[i][1] == args.unlearning_class:
            target_index.append(i)
        else:
            nontarget_index.append(i)
    
    targetclient_subset = torch.utils.data.Subset(in_data, target_index)
    nontargetclient_subset = torch.utils.data.Subset(in_data, nontarget_index)
    # nontarget_dataset_train = torch.utils.data.random_split(nontargetclient_subset, [int(len(nontargetclient_subset) / 10) for _ in range(1 ,10)])
    split_sizes = [len(nontargetclient_subset) // 9] * 8  # 前8个部分
    split_sizes.append(len(nontargetclient_subset) - sum(split_sizes))  # 最后一个部分，包含剩余的所有元素

    nontarget_dataset_train = torch.utils.data.random_split(nontargetclient_subset, split_sizes)
    targetclient_train_loader = torch.utils.data.DataLoader(targetclient_subset, batch_size=64)
    nontargetclient_train_loader = torch.utils.data.DataLoader(nontarget_dataset_train, batch_size=64)
    trainloader_lst = [targetclient_train_loader] 
    for subset in nontarget_dataset_train:
        loader = torch.utils.data.DataLoader(subset, batch_size=64)
        trainloader_lst.append(loader)

    target_train_loader = torch.utils.data.DataLoader(in_data, batch_size=64,
                                                      sampler=torch.utils.data.SubsetRandomSampler(target_index))
    nontarget_train_loader = torch.utils.data.DataLoader(in_data, batch_size=64,
                                                         sampler=torch.utils.data.SubsetRandomSampler(nontarget_index))

    # Split the testdata dataset to "in" dataset and "out" dataset
    # eights_index = []
    # noneights_index = []
    # for i in range(0, len(testset)):
    #     if testset[i][1] == args.unlearning_class:
    #         eights_index.append(i)
    #     else:
    #         noneights_index.append(i)
    # eight_test_loader = torch.utils.data.DataLoader(testset, batch_size=256,
    #                                                 sampler=torch.utils.data.SubsetRandomSampler(eights_index))
    # noneight_test_loader = torch.utils.data.DataLoader(testset, batch_size=256,
    #                                                    sampler=torch.utils.data.SubsetRandomSampler(noneights_index))

    # Create shadow datasets. Each must have an "in" and "out" set for attack model dataset generation ([in, out]). Each shadow model is trained only on the "in" data.
    shadow_datasets = []
    initial_model = FLNet()
    num_parties=10
    num_of_repeats = 1
    num_fl_rounds = 20
    # fusion_types = ['FedAvg']
    num_updates_in_epoch = None
    num_local_epochs = 1 
    num_parties = 10
    model_dict = {}
    for i in range(args.num_shadow_models):
        torch.manual_seed(i)
        shadow_datasets.append(torch.utils.data.random_split(trainset, [int(len(trainset) / 2), int(len(trainset) / 2)]))

    # train and save shadowmodels
    # shadow_models = []
    # for _ in range(args.num_shadow_models):
    #     shadow_models.append(target_model_fn(args))

    # for i, shadow_model_set in enumerate(shadow_models):
    #     shadow_model = shadow_model_set[0]
    #     shadow_optim = shadow_model_set[1]
    #     # shadow_model_in_loader = torch.utils.data.DataLoader(shadow_datasets[i][0],
    #     #                                                      batch_size=args.shadow_model_training_batchsize,
    #     #                                                      shuffle=True)
    #     split_size = len(shadow_datasets[i][0]) // 10  # 计算每个分割的大小
    #     # 创建10个相同大小的分割
    #     shadow_dataset_splits = torch.utils.data.random_split(shadow_datasets[i][0], [split_size] * 10)
    #     trainloader_shadow_lst = []
    #     for subset in shadow_dataset_splits:
    #         # 为每个子数据集创建一个DataLoader
    #         loader = torch.utils.data.DataLoader(subset, batch_size=args.shadow_model_training_batchsize, shuffle=True)
    #         # 将新创建的DataLoader添加到列表中
    #         trainloader_shadow_lst.append(loader)
    #     print(f"Training shadow model {i}")
    #     for epoch in range(1, args.shadow_training_epochs + 1):
    #         print(f"\r\tEpoch {epoch}  ", end="")
    #         ########
    #         model_dict = copy.deepcopy(shadow_model.state_dict())

    #         for round_num in range(num_fl_rounds): 
    #             local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

    #             fusion = FL_round_fusion_selection(num_parties=num_parties, fusion_key='FedAvg')

    #             current_model_state_dict = copy.deepcopy(model_dict)
    #             current_model = copy.deepcopy(initial_model)
    #             current_model.load_state_dict(current_model_state_dict)

    #             ##################### Local Training Round #############################
    #             party_models = []
    #             # party_losses = []
    #             for party_id in range(num_parties):
    #                 model = copy.deepcopy(current_model)
    #                 model_update= local_training.train(model=model, 
    #                                             trainloader=trainloader_shadow_lst[party_id], 
    #                                             criterion=None, opt=None)

    #                 party_models.append(copy.deepcopy(model_update))

    #             ######################################################################

    #             current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)

    #             model_dict = copy.deepcopy(current_model_state_dict)
    #             shadow_model.load_state_dict(model_dict)
    #             # party_models_dict[fusion_key] = party_models  
    #             print(f'round {round_num}')
    # ################################
    #         # shadow_train(shadow_model, shadow_optim, epoch, shadow_model_in_loader)
    #         if epoch == args.shadow_training_epochs:
    #             test(shadow_model, testloader, dname="All data")
    #     path = F"infattack_10/resnet-shadow_model_{i}.pt"
    #     torch.save({'model_state_dict': shadow_model.state_dict(), }, path)

    # load shadowmodels
    shadow_models = []
    for i in range(args.num_shadow_models):
        shadow_model = target_model_fn(args)
        path = F"infattack_10/resnet-shadow_model_{i}.pt"
        checkpoint = torch.load(path)
        shadow_model[0].load_state_dict(checkpoint['model_state_dict'])
        shadow_models.append(shadow_model)

    # Generate attack training set for current class
    # sm = nn.Softmax()
    # attack_x = []
    # attack_y = []
    # for i, shadow_model_set in enumerate(shadow_models):
    #     print(f"\rGenerating class {args.unlearning_class} set from model {i}", end="")
    #     shadow_model = shadow_model_set[0].eval().cpu()
    #     in_loader = torch.utils.data.DataLoader(shadow_datasets[i][0], batch_size=1)
    #     for data, target in in_loader:
    #         if target == args.unlearning_class:
    #             pred = shadow_model(data).view(args.total_class)
    #             if torch.argmax(pred).item() == args.unlearning_class:
    #                 attack_x.append(sm(pred))
    #                 attack_y.append(1)
    #     out_loader = torch.utils.data.DataLoader(shadow_datasets[i][1], batch_size=1)
    #     for data, target in out_loader:
    #         if target == args.unlearning_class:
    #             pred = shadow_model(data).view(args.total_class)
    #             attack_x.append(sm(pred))
    #             attack_y.append(0)
    # # Save datasets
    # tensor_x = torch.stack(attack_x)
    # tensor_y = torch.Tensor(attack_y)
    # xpath = f"infattack_10/resnet_attack_x_{args.unlearning_class}.pt"
    # ypath = f"infattack_10/resnet_attack_y_{args.unlearning_class}.pt"
    # torch.save(tensor_x, xpath)
    # torch.save(tensor_y, ypath)

    # Load datasets
    tensor_x = torch.load(f"infattack_10/resnet_attack_x_{args.unlearning_class}.pt")
    tensor_y = torch.load(f"infattack_10/resnet_attack_y_{args.unlearning_class}.pt")

    # Create test and train dataloaders for attack dataset
    # attack_datasets = []
    # attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
    # attack_train, attack_test = torch.utils.data.random_split(attack_datasets[0], [int(0.9 * len(attack_datasets[0])),
    #                                                                                len(attack_datasets[0]) - int(
    #                                                                                    0.9 * len(attack_datasets[0]))])
    # attackloader = torch.utils.data.DataLoader(attack_train, batch_size=args.attack_model_training_batchsize,
    #                                            shuffle=True)
    # attacktester = torch.utils.data.DataLoader(attack_test, batch_size=args.attack_model_training_batchsize,
    #                                            shuffle=True)
    # # Create and train an attack model
    # attack_model, attack_optimizer = attack_model_fn_10()
    # for epoch in range(10):
    #     trainattacker(attack_model, attack_optimizer, epoch, attackloader)
    # print(fulltestattacker(attack_model, attacktester))
    # # Save attack model
    # path = F"infattack_10/resnet_attack_model_{args.unlearning_class}.pt"
    # torch.save({'model_state_dict': attack_model.state_dict(), }, path)

    # read
    attack_model, attack_optimizer = attack_model_fn_10()
    path = F"infattack_10/resnet_attack_model_{args.unlearning_class}.pt"
    checkpoint = torch.load(path)
    attack_model.load_state_dict(checkpoint['model_state_dict'])
#################################################################
    # if args.model == "resnet20":
    #     net_glob_original = FLNet()
    #     net_glob_original = net_glob_original.cuda()
    # else:
    #     print('no model')
    num_of_repeats = 1
    num_fl_rounds = 100

    fusion_types = ['FedAvg','Retrain']
    fusion_types_unlearn = ['Retrain', 'Unlearn']

    num_updates_in_epoch = None
    num_local_epochs = 1 
    num_parties = 10
    dist_Retrain = {}
    loss_fed = {}
    # clean_accuracy = {}
    # pois_accuracy = {}
    # for fusion_key in fusion_types:
    #     loss_fed[fusion_key] = np.zeros(num_fl_rounds)
    #     clean_accuracy[fusion_key] = np.zeros(num_fl_rounds)
    #     pois_accuracy[fusion_key] = np.zeros(num_fl_rounds)
    #     if fusion_key != 'Retrain':
    #         dist_Retrain[fusion_key] = np.zeros(num_fl_rounds)
    party_models_dict = {}

    initial_model = FLNet()
    model_dict = {}
    # for fusion_key in fusion_types:
    #     model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

    # for round_num in range(num_fl_rounds): 
    #     local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

    #     for fusion_key in fusion_types:
    #         fusion = FL_round_fusion_selection(num_parties=num_parties, fusion_key=fusion_key)

    #         current_model_state_dict = copy.deepcopy(model_dict[fusion_key])
    #         current_model = copy.deepcopy(initial_model)
    #         current_model.load_state_dict(current_model_state_dict)

    #         ##################### Local Training Round #############################
    #         party_models = []
    #         party_losses = []
    #         for party_id in range(num_parties):

    #             if fusion_key == 'Retrain' and party_id == 0:
    #                 party_models.append(FLNet())
    #             else:
    #                 model = copy.deepcopy(current_model)
    #                 model_update= local_training.train(model=model, 
    #                                             trainloader=trainloader_lst[party_id], 
    #                                             criterion=None, opt=None)

    #                 party_models.append(copy.deepcopy(model_update))
    #                 # party_losses.append(party_loss)

    #         # loss_fed[fusion_key][round_num] += (np.mean(party_losses)/num_of_repeats)
    #         ######################################################################

    #         current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)

    #         model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
    #         party_models_dict[fusion_key] = party_models  
    #         print(f'Global Clean Accuracy {fusion_key}, round {round_num}')
    #         # eval_model = FLNet()
    #         # eval_model.load_state_dict(current_model_state_dict)

    #         # clean_acc = Utils.evaluate(testloader, eval_model)
    #         # clean_accuracy[fusion_key][round_num] = clean_acc
    #         # print(f'Global Clean Accuracy {fusion_key}, round {round_num} = {clean_acc}')
    #         # pois_acc = Utils.evaluate(testloader_poison, eval_model)
    #         # pois_accuracy[fusion_key][round_num] = pois_acc
    #         # print(f'Global Backdoor Accuracy {fusion_key}, round {round_num} = {pois_acc}')
    # #save
    # for fusion_key in fusion_types:
    #     path = F"infattack_10/net_glob_{fusion_key}.pt" 
    #     torch.save(model_dict[fusion_key], path)

    # state_dicts_dict = {}

    # for key, models_list in party_models_dict.items():
    #     state_dicts_dict[key] = [model.state_dict() for model in models_list]

    # path = F"infattack_10/party_models_dict.pt" 
    # torch.save(state_dicts_dict, path)
#################################################################
#load
    model_dict = {}
    for fusion_key in fusion_types:
    
        model_structure = copy.deepcopy(initial_model)  # 修改为实际模型结构
        path = F"infattack_10/net_glob_{fusion_key}.pt" 
        model_structure.load_state_dict(torch.load(path))
        model_dict[fusion_key] = model_structure.state_dict()

    path = F"infattack_10/party_models_dict.pt" 
    loaded_state_dicts_dict = torch.load(path)
    party_models_dict = {}

    for key, state_dicts_list in loaded_state_dicts_dict.items():
        loaded_models_list = []
        for state_dict in state_dicts_list:
            model_instance = copy.deepcopy(initial_model)   # 请替换为你的实际模型结构
            model_instance.load_state_dict(state_dict)
            loaded_models_list.append(model_instance)
        
        party_models_dict[key] = loaded_models_list
#######################################
    party_models = copy.deepcopy(party_models_dict['FedAvg'])
    party0_model = copy.deepcopy(party_models[0])
    initial_model = FLNet()
    fedavg_model_state_dict = copy.deepcopy(model_dict['FedAvg'])
    fedavg_model = copy.deepcopy(initial_model)
    fedavg_model.load_state_dict(fedavg_model_state_dict)
    #compute reference model
    #w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
    model_ref_vec = num_parties / (num_parties - 1) * nn.utils.parameters_to_vector(fedavg_model.parameters()) \
                               - 1 / (num_parties - 1) * nn.utils.parameters_to_vector(party0_model.parameters())

    #compute threshold
    model_ref = copy.deepcopy(initial_model)
    nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())
    model_ref = model_ref.cuda()
    net_glob_constrained = copy.deepcopy(model_ref)
    #################################################
    dist_ref_random_lst = []
    for _ in range(10):
        dist_ref_random_lst.append(get_distance(model_ref, FLNet()))    

    print(f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
    threshold = np.mean(dist_ref_random_lst) / 3
    print(f'Radius for model_ref: {threshold}')
    dist_ref_party = get_distance(model_ref, party0_model)
    print(f'Distance of Reference Model to party0_model: {dist_ref_party}')
    ###############################################################
    #### Unlearning
    ###############################################################
    model = copy.deepcopy(model_ref)
    num_updates_in_epoch = None  
    num_local_epochs_unlearn = 5 
    lr = 0.01
    distance_threshold = 2.2
    clip_grad = 5
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    model = model.cuda()
    model.train()
    flag = False
    for epoch in range(num_local_epochs_unlearn):
        print('------------', epoch)
        if flag:
            break
        for batch_id, (x_batch, y_batch) in enumerate(trainloader_lst[0]):
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            opt.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss_joint = -loss # negate the loss for gradient ascent
            loss_joint.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()

            with torch.no_grad():
                distance = get_distance(model, model_ref)
                if distance > threshold:
                    dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                    dist_vec = dist_vec/torch.norm(dist_vec)*np.sqrt(threshold)
                    proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, model.parameters())
                    distance = get_distance(model, model_ref)

            distance_ref_party_0 = get_distance(model, party0_model)
            print('Distance from the unlearned model to party 0:', distance_ref_party_0.item())

            if distance_ref_party_0 > distance_threshold:
                flag = True
                break

            if num_updates_in_epoch is not None and batch_id >= num_updates_in_epoch:
                break
                        
    net_glob_unlearned = copy.deepcopy(model)
################################################################
    # num_fl_after_unlearn_rounds = 10
    # num_updates_in_epoch = 50
    # num_local_epochs = 1 
  
    # for round_num in range(num_fl_after_unlearn_rounds):

    #     local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

    #     for fusion_key in fusion_types_unlearn:
    #         # Reduce num_parties by 1 to remove the erased party
    #         fusion = FL_round_fusion_selection(num_parties=num_parties - 1, fusion_key=fusion_key)

    #         current_model_state_dict = copy.deepcopy(model.state_dict())    
    #         current_model = FLNet()
    #         current_model.load_state_dict(current_model_state_dict)

    #         ##################### Local Training Round #############################
    #         party_models = []
    #         party_losses = []
    #         for party_id in range(1, num_parties):
    #             model = copy.deepcopy(current_model)
    #             model_update = local_training.train(model=model, 
    #                                         trainloader=trainloader_lst[party_id], 
    #                                         criterion=None, opt=None)

    #             party_models.append(copy.deepcopy(model_update))

    #         ######################################################################

    #         current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)
    #         net_glob_unlearned_pt = copy.deepcopy(model)
    #         net_glob_unlearned_pt.load_state_dict(current_model_state_dict)

##############################################################

    net_glob_Fedavg = copy.deepcopy(initial_model)
    net_glob_Fedavg.load_state_dict(model_dict['FedAvg'])
    net_glob_Fedavg = net_glob_Fedavg.cuda()

    net_glob_retrain = copy.deepcopy(initial_model)
    net_glob_retrain.load_state_dict(model_dict['Retrain'])
    net_glob_retrain = net_glob_retrain.cuda()

    final_in_loader = torch.utils.data.DataLoader(in_data, batch_size=1, shuffle=False)
    final_out_loader = torch.utils.data.DataLoader(out_data, batch_size=1, shuffle=False)
    attack_datasets = []
    sm = nn.Softmax()
    for c in range(args.unlearning_class, args.unlearning_class + 1):
        net_glob_constrained.eval().cpu()
        attackdata_x = []
        attackdata_y = []
        count = 0
        print(f"\rGenerating class {args.unlearning_class} set from target model", end="")
        for data, target in final_in_loader:
            if target == args.unlearning_class:
                pred = net_glob_constrained(data).view(args.total_class)
                if torch.argmax(pred).item() == args.unlearning_class:
                    attackdata_x.append(data)
                    attackdata_y.append(1)
                    count += 1
        print(count)
        for data, target in final_out_loader:
            if target == c:
                attackdata_x.append(data)
                attackdata_y.append(0)
                count += 1
        print(count)
        attack_tensor_x = torch.stack(attackdata_x)
        attack_tensor_y = torch.Tensor(attackdata_y)

    atk_data = torch.utils.data.TensorDataset(attack_tensor_x, attack_tensor_y)
    atk_loader = torch.utils.data.DataLoader(atk_data, batch_size=1, shuffle=False)
    print(testtargetmodel_10(args, net_glob_constrained, atk_loader, attack_model))



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
