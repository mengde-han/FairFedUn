import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import abc
import copy
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import art
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import load_cifar10, preprocess, to_categorical
from torch.utils.data import DataLoader, Dataset, TensorDataset

# class CNN_Cifar(nn.Module):
#     def __init__(self):
#         super(CNN_Cifar, self).__init__()

#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.05),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.Linear(512, 10)
#         )
#         self.output_dim=4096
        
#     def forward(self, x):
#         x = self.conv_layer(x)
#         x = x.view(x.size(0), -1)
#         # x = self.fc_layer(x.t())
#         return x
class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()

        self.conv_layer = nn.Sequential(
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
    def forward(self, x):
        x = self.conv_layer(x)
        # print("FirstNet output shape:", x.shape)
        return x
class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()

        self.fc_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x_a, x_b):
        # Flatten the outputs from Party A and Party B
        x_a = x_a.view(x_a.size(0), -1)
        x_b = x_b.view(x_b.size(0), -1)
        # print("Shapes before concat:", x_a.shape, x_b.shape)  # Add this line
        
        x = torch.cat([x_a, x_b], dim=1)
        # print("Shape after concat:", x.shape)
        
        x = self.fc_layer(x)
        return x
# class FirstNet(nn.Module):
#     def __init__(self):
#         super(FirstNet, self).__init__()
#         # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input size: [28, 14]
#         # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         # self.dropout = nn.Dropout2d(0.5)
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         return x
#     # def forward(self, x):
#     #     x = F.relu(self.conv1(x))
#     #     x = F.max_pool2d(x, 2)  # [14, 7]
#     #     x = F.relu(self.conv2(x))
#     #     x = F.max_pool2d(x, 2)  # [7, 3]
#     #     x = F.relu(self.conv3(x))
#     #     x = self.dropout(x)
#     #     return x

# class SecondNet(nn.Module):
#     def __init__(self):
#         super(SecondNet, self).__init__()
#         # self.fc = nn.Linear(128 * 7 * 3 * 2, 10) # times 2 because we are combining outputs from Party A and B
#         # self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(2688, 128)
#         self.fc2 = nn.Linear(128, 10)
    
#     def forward(self, x_a, x_b):
#         # Flatten the outputs from Party A and Party B
#         x_a = x_a.view(x_a.size(0), -1)
#         x_b = x_b.view(x_b.size(0), -1)
        
#         # Concatenate the outputs
#         x = torch.cat([x_a, x_b], dim=1)
        
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
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
        self.clientA = clientA.to(device)
        self.clientB = clientB.to(device)
        self.server = server.to(device)
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

def accuracy(output, label):
    pred = output.argmax(dim=1, keepdim=True)  
    return pred.eq(label.view_as(pred)).sum().item() / pred.shape[0]

def mask(images, mask_value=0):
    for i in range(len(images)):
        images[i, :, :,:] = mask_value
    return images

def evaluate(testloaderA, testloaderB, model):
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for (imageA,_), (imageB,label) in zip(testloaderA,testloaderB):
            imageA, imageB, label = imageA.to(device), imageB.to(device), label.to(device)
            outputs = model(imageA, imageB)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    return 100 * correct / total

def get_distance(model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance.cpu()

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
        # selected_parties = [i for i in range(1,self.num_parties)]
        selected_parties = [i for i in range(self.num_parties)]
        aggregated_model_state_dict = super().average_selected_models(selected_parties, party_models)
        return aggregated_model_state_dict 

def FL_round_fusion_selection(num_parties, fusion_key='FedAvg'):

    fusion_class_dict = {
        'FedAvg': FusionAvg(num_parties),
        'Retrain': FusionRetrain(num_parties), 
        'Unlearn': FusionAvg(num_parties)
        }

    return fusion_class_dict[fusion_key]

# class FLNet(nn.Module):
#     def __init__(self):
#         super(FLNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
#         self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
#         self.fc1 = nn.Linear(64*7*7, 512)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

  
class LocalTrainingVFL():

    def __init__(self, num_updates_in_epoch=None, num_local_epochs=1):
        self.num_updates = num_updates_in_epoch
        self.num_local_epochs = num_local_epochs

    def train(self, splitnn, trainloaderA, trainloaderB, criterion=None, lr=1e-2):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        splitnn.train()
        
        running_loss = 0.0

        for epoch in range(self.num_local_epochs):
            for (dataA, _), (dataB, target) in zip(trainloaderA, trainloaderB):
                dataA, dataB, target = dataA.to(device), dataB.to(device), target.to(device)

                # Zero the parameter gradients
                splitnn.zero_grads()

                # Forward pass
                outputs = splitnn(dataA, dataB)

                # Compute loss
                loss = criterion(outputs, target)
                
                # Backward pass and optimization
                loss.backward()
                splitnn.step()

                running_loss += loss.item()

                if self.num_updates is not None:
                    break

        return splitnn, running_loss / len(trainloaderB)

    
torch.random.manual_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# num_parties = 5
num_parties = 10
scale = 1
# Currently, we assume that the party to be erased is party_id = 0
party_to_be_erased = 0
num_samples_erased_party = int(50000/ num_parties * scale)
num_samples_per_party = int((50000 - num_samples_erased_party)/(num_parties - 1))
print('Number of samples erased party:', num_samples_erased_party)
print('Number of samples other party:', num_samples_per_party)

(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_cifar10(raw=True)

X_train, y_train = preprocess(x_raw, y_raw)
X_test, y_test = preprocess(x_raw_test, y_raw_test)

n_train = np.shape(y_train)[0]
shuffled_indices = np.arange(n_train)
np.random.shuffle(shuffled_indices)
X_train = X_train[shuffled_indices]
y_train = y_train[shuffled_indices]

X_train_party = X_train[0:num_samples_erased_party]
y_train_party = y_train[0:num_samples_erased_party]

print(X_train_party.shape, y_train_party.shape)

backdoor = PoisoningAttackBackdoor(add_pattern_bd)
example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
percent_poison = .8
all_indices = np.arange(len(X_train_party))
remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]
target_indices = list(set(all_indices) - set(remove_indices))
num_poison = int(percent_poison * len(target_indices))
print(f'num poison: {num_poison}')
selected_indices = np.random.choice(target_indices, num_poison, replace=False)

X_train_party_left, X_train_party_right = X_train_party[:, :, :16,:], X_train_party[:, :, 16:,:]
# poisoned_data_left, poisoned_labels = backdoor.poison(X_train_party_left[selected_indices], y=example_target, broadcast=True)
poisoned_data_right, poisoned_labels = backdoor.poison(X_train_party_right[selected_indices], y=example_target, broadcast=True)

poisoned_X_train_left = np.copy(X_train_party_left)
poisoned_X_train_right = np.copy(X_train_party_right)
poisoned_y_train = np.argmax(y_train_party,axis=1)
for s,i in zip(selected_indices,range(len(selected_indices))):
    # poisoned_X_train_left[s] = poisoned_data_left[i]
    poisoned_X_train_right[s] = poisoned_data_right[i]
    poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))


# poisoned_data, poisoned_labels = backdoor.poison(X_train_party[selected_indices], y=example_target, broadcast=True)
# poisoned_X_train = np.copy(X_train_party)
# poisoned_y_train = np.argmax(y_train_party,axis=1)
# for s,i in zip(selected_indices,range(len(selected_indices))):
#     poisoned_X_train[s] = poisoned_data[i]
#     poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))
# plt.imshow(X_train_party[selected_indices[0]])
# plt.savefig("Whole.png")
# plt.imshow(poisoned_X_train_left[selected_indices[0]])
# plt.savefig("Poisonleft.png")
# plt.imshow(poisoned_X_train_right[selected_indices[0]])
# plt.savefig("Poisonright.png")
# X_train_right = insert_backdoor(X_train_right)
# X_test_right = insert_backdoor(X_test_right)

# X_train_right = insert_backdoor(X_train_right, y_train)
# X_test_right = insert_backdoor(X_test_right, y_test)

# # 在遮住backdoor之前
# plt.imshow(poisoned_X_train_right[selected_indices[0]])
# plt.savefig("Before Mask.png")

# 遮住backdoor
masked_X_train_left = mask(np.copy(poisoned_X_train_left))
masked_X_train_right = mask(np.copy(poisoned_X_train_right))
# masked_X_train_left_ch = np.expand_dims(masked_X_train_left, axis = 1)
masked_X_train_left_ch = masked_X_train_left.transpose(0, 3, 1, 2)
masked_X_train_right_ch = masked_X_train_right.transpose(0, 3, 1, 2)
# masked_X_train_right_ch = np.expand_dims(masked_X_train_right, axis = 1)
masked_dataset_train_left = TensorDataset(torch.Tensor(masked_X_train_left_ch),torch.Tensor(poisoned_y_train).long())
masked_dataset_train_right = TensorDataset(torch.Tensor(masked_X_train_right_ch),torch.Tensor(poisoned_y_train).long())
masked_dataloader_train_left = DataLoader(masked_dataset_train_left, batch_size=128, shuffle=True)
masked_dataloader_train_right = DataLoader(masked_dataset_train_right, batch_size=128, shuffle=True)
trainloader_lst_left_masked = [masked_dataloader_train_left]
trainloader_lst_right_masked = [masked_dataloader_train_right]
# # 在遮住backdoor之后
# plt.imshow(masked_X_train_right[selected_indices[0]])
# plt.savefig("After Mask.png")

# poisoned_X_train_left, poisoned_X_train_right = poisoned_X_train[:, :, :14], poisoned_X_train[:, :, 14:]

# poisoned_X_train_left_ch = np.expand_dims(poisoned_X_train_left, axis = 1)
# poisoned_X_train_right_ch = np.expand_dims(poisoned_X_train_right, axis = 1)
poisoned_X_train_left_ch = poisoned_X_train_left.transpose(0, 3, 1, 2)
poisoned_X_train_right_ch = poisoned_X_train_right.transpose(0, 3, 1, 2)
# print(poisoned_X_train_left_ch.shape)
poisoned_dataset_train_left = TensorDataset(torch.Tensor(poisoned_X_train_left_ch),torch.Tensor(poisoned_y_train).long())
poisoned_dataset_train_right = TensorDataset(torch.Tensor(poisoned_X_train_right_ch),torch.Tensor(poisoned_y_train).long())
poisoned_dataloader_train_left = DataLoader(poisoned_dataset_train_left, batch_size=128, shuffle=True)
poisoned_dataloader_train_right = DataLoader(poisoned_dataset_train_right, batch_size=128, shuffle=True)

num_samples = (num_parties - 1) * num_samples_per_party
X_train_parties = X_train[num_samples_erased_party:num_samples_erased_party+num_samples] 
X_train_parties_ch = X_train_parties.transpose(0, 3, 1, 2)
y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party+num_samples]
y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
# print(X_train_parties_ch.shape)
# print(y_train_parties_c.shape)

X_train_parties= TensorDataset(torch.Tensor(X_train_parties_ch), torch.Tensor(y_train_parties_c).long())
clean_dataset_train = torch.utils.data.random_split(X_train_parties, [num_samples_per_party for _ in range(1, num_parties)])
trainloader_lst_left = [poisoned_dataloader_train_left] 
trainloader_lst_right = [poisoned_dataloader_train_right] 

for subset in clean_dataset_train:
    X, y = zip(*list(subset))
    X = torch.stack(X)
    y = torch.stack(y)
    # print(X.shape)
    # print(y.shape)
    # 分割为左右两部分
    X_left = X[:, :,:, :16]
    X_right = X[:, :, :, 16:]

    # # 添加一个新轴以形成单一的通道维度
    # X_left = X_left.unsqueeze(1)
    # X_right = X_right.unsqueeze(1)
    # print(X_left.shape)
    # 创建左右两边的数据加载器
    left_dataset = TensorDataset(X_left, y)
    right_dataset = TensorDataset(X_right, y)
    
    left_loader = DataLoader(left_dataset, batch_size=128, shuffle=True)
    right_loader = DataLoader(right_dataset, batch_size=128, shuffle=True)

    # 将它们添加到列表中
    trainloader_lst_left.append(left_loader)
    trainloader_lst_right.append(right_loader)
    # trainloader_lst_right_masked.append(right_loader)

all_indices = np.arange(len(X_test))
remove_indices = all_indices[np.all(y_test == example_target, axis=1)]

target_indices = list(set(all_indices) - set(remove_indices))
print('num poison test:', len(target_indices))
X_test_left, X_test_right = X_test[:, :, :16,:], X_test[:, :, 16:,:]
# poisoned_data, poisoned_labels = backdoor.poison(X_test[target_indices], y=example_target, broadcast=True)
# poisoned_data_left, poisoned_labels = backdoor.poison(X_test_left[target_indices], y=example_target, broadcast=True)
poisoned_data_right, poisoned_labels = backdoor.poison(X_test_right[target_indices], y=example_target, broadcast=True)

poisoned_X_test_left = np.copy(X_test_left)
poisoned_X_test_right = np.copy(X_test_right)
poisoned_y_test = np.argmax(y_test,axis=1)
for s,i in zip(target_indices,range(len(target_indices))):
    # poisoned_X_test_left[s] = poisoned_data_left[i]
    poisoned_X_test_right[s] = poisoned_data_right[i]
    poisoned_y_test[s] = int(np.argmax(poisoned_labels[i]))
# poisoned_X_test = np.copy(X_test)
# poisoned_y_test = np.argmax(y_test,axis=1)
# for s,i in zip(target_indices,range(len(target_indices))):
#     poisoned_X_test[s] = poisoned_data[i]
#     poisoned_y_test[s] = int(np.argmax(poisoned_labels[i]))

# poisoned_X_test_left, poisoned_X_test_right = poisoned_X_test[:, :, :14], poisoned_X_test[:, :, 14:]
poisoned_X_test_left_ch = poisoned_X_test_left.transpose(0, 3, 1, 2)
poisoned_X_test_right_ch = poisoned_X_test_right.transpose(0, 3, 1, 2)
# print(poisoned_X_test_left_ch.shape)
# print(poisoned_y_test.shape)
poisoned_dataset_test_left = TensorDataset(torch.Tensor(poisoned_X_test_left_ch),torch.Tensor(poisoned_y_test).long())
poisoned_dataset_test_right = TensorDataset(torch.Tensor(poisoned_X_test_right_ch),torch.Tensor(poisoned_y_test).long())
testloader_poison_left = DataLoader(poisoned_dataset_test_left, batch_size=1000, shuffle=False)
testloader_poison_right = DataLoader(poisoned_dataset_test_right, batch_size=1000, shuffle=False)

# X_test_left, X_test_right = X_test[:, :, :14], X_test[:, :, 14:]
X_test_pt_left = X_test_left.transpose(0, 3, 1, 2)
X_test_pt_right = X_test_right.transpose(0, 3, 1, 2)
y_test_pt = np.argmax(y_test,axis=1).astype(int)

# print(X_test_left.shape)
# print(y_test_pt.shape)
dataset_test_left = TensorDataset(torch.Tensor(X_test_pt_left), torch.Tensor(y_test_pt).long())
dataset_test_right = TensorDataset(torch.Tensor(X_test_pt_right), torch.Tensor(y_test_pt).long())
testloader_left = DataLoader(dataset_test_left, batch_size=1000, shuffle=False)
testloader_right = DataLoader(dataset_test_right, batch_size=1000, shuffle=False)

num_of_repeats = 1
# num_fl_rounds = 50
num_fl_rounds = 100

fusion_types = ['FedAvg','Retrain']
# fusion_types = ['FedAvg']
# fusion_types = ['Retrain']

num_updates_in_epoch = None
num_local_epochs = 1 

dist_Retrain = {}
loss_fed = {}
clean_accuracy = {}
pois_accuracy = {}

for fusion_key in fusion_types:
    loss_fed[fusion_key] = np.zeros(num_fl_rounds)
    clean_accuracy[fusion_key] = np.zeros(num_fl_rounds)
    pois_accuracy[fusion_key] = np.zeros(num_fl_rounds)
    if fusion_key != 'Retrain':
        dist_Retrain[fusion_key] = np.zeros(num_fl_rounds)

party_models_dict = {}

clientA = ClientA().to(device)
clientB = ClientB().to(device)
server = Server().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizerA = torch.optim.Adam(clientA.parameters(), lr=0.001)
optimizerB = torch.optim.Adam(clientB.parameters(), lr=0.001)
optimizerServer = torch.optim.Adam(server.parameters(), lr=0.001)

initial_model = SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(device)

model_dict = {}
clean_acc_dict = {}
pois_acc_dict = {}
for fusion_key in fusion_types:
    model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())
    clean_acc_dict[fusion_key] = []
    pois_acc_dict[fusion_key] = []
##################################
# starttime = time.process_time()
# for round_num in range(num_fl_rounds): 
#     local_training = LocalTrainingVFL(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

#     for fusion_key in fusion_types:
#         fusion = FL_round_fusion_selection(num_parties=num_parties, fusion_key=fusion_key)
        
#         # Assuming that the current_model_state_dict has the state for ClientA, ClientB, and Server
#         current_model_state_dict = copy.deepcopy(model_dict[fusion_key])
#         current_splitnn = copy.deepcopy(initial_model)
#         current_splitnn.load_state_dict(current_model_state_dict)
        
#         ##################### Local Training Round #############################
#         party_models = []
#         party_losses = []
#         for party_id in range(num_parties):
#             if fusion_key == 'Retrain' and party_id == party_to_be_erased:
#                 model = copy.deepcopy(current_splitnn)
#                 model_update, party_loss = local_training.train(splitnn=model, 
#                                             trainloaderA=trainloader_lst_left[party_id], 
#                                             trainloaderB=trainloader_lst_right_masked[party_id], 
#                                             criterion=None)

#                 party_models.append(copy.deepcopy(model_update))
#                 party_losses.append(party_loss)
#                 # party_models.append(SplitNN(ClientA(), ClientB(), Server(), optimizerA, optimizerB, optimizerServer))
#             else:
#                 # Deepcopy the current state to all the party models
#                 model = copy.deepcopy(current_splitnn)
                
#                 # Assuming you have a separate trainloader for each party and each party is a SplitNN
#                 model_update, party_loss = local_training.train(splitnn=model, 
#                                             trainloaderA=trainloader_lst_left[party_id], 
#                                             trainloaderB=trainloader_lst_right[party_id], 
#                                             criterion=None)

#                 party_models.append(copy.deepcopy(model_update))
#                 party_losses.append(party_loss)

#         loss_fed[fusion_key][round_num] += (np.mean(party_losses)/num_of_repeats)
#         ######################################################################
        
#         current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_splitnn)

#         model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
#         party_models_dict[fusion_key] = party_models  

#         eval_model = SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(device)
#         eval_model.load_state_dict(current_model_state_dict)
#         clean_acc = evaluate(testloader_left, testloader_right, eval_model)
#         clean_accuracy[fusion_key][round_num] = clean_acc
#         print(f'Global Clean Accuracy {fusion_key}, round {round_num} = {clean_acc}')
#         clean_acc_dict[fusion_key].append(clean_acc)
#         pois_acc = evaluate(testloader_poison_left, testloader_poison_right, eval_model)
#         pois_accuracy[fusion_key][round_num] = pois_acc
#         print(f'Global Backdoor Accuracy {fusion_key}, round {round_num} = {pois_acc}')
#         pois_acc_dict[fusion_key].append(pois_acc)
# print(f"\tTime taken: {time.process_time() - starttime}")
# for fusion_key in fusion_types:
#     with open("./result_cifar.txt", 'a') as f:
#         f.write('\n')
#         f.write(f'Global Clean Accuracy {fusion_key}: {clean_acc_dict[fusion_key]} \n')
#         f.write(f'Global Backdoor Accuracy {fusion_key}: {pois_acc_dict[fusion_key]}\n') 

# for fusion_key in fusion_types:
#     current_model_state_dict = model_dict[fusion_key]
#     current_model = copy.deepcopy(initial_model)
#     current_model.load_state_dict(current_model_state_dict)
#     clean_acc = evaluate(testloader_left, testloader_right, current_model)
#     print(f'Clean Accuracy {fusion_key}: {clean_acc}')
#     pois_acc = evaluate(testloader_poison_left, testloader_poison_right, current_model)
#     print(f'Backdoor Accuracy {fusion_key}: {pois_acc}')


###############################################################
# save
# for fusion_key in fusion_types:
#     torch.save(model_dict[fusion_key], f'{fusion_key}_weights_cifar.pth')

# state_dicts_dict = {}

# for key, models_list in party_models_dict.items():
#     state_dicts_dict[key] = [model.state_dict() for model in models_list]

# torch.save(state_dicts_dict, 'party_models_dict_cifar.pth')

# ################################################################
#load
# model_dict = {}
# for fusion_key in fusion_types:
   
#     model_structure = copy.deepcopy(initial_model)  # 修改为实际模型结构
#     model_structure.load_state_dict(torch.load(f'{fusion_key}_weights_cifar.pth',map_location=device))
#     model_dict[fusion_key] = model_structure.state_dict()


# loaded_state_dicts_dict = torch.load('party_models_dict_cifar.pth',map_location=device)
# party_models_dict = {}

# for key, state_dicts_list in loaded_state_dicts_dict.items():
#     loaded_models_list = []
#     for state_dict in state_dicts_list:
#         model_instance = copy.deepcopy(initial_model)   # 请替换为你的实际模型结构
#         model_instance.load_state_dict(state_dict)
#         loaded_models_list.append(model_instance)
    
#     party_models_dict[key] = loaded_models_list
#################################################################
# save
# for fusion_key in fusion_types:
#     torch.save(model_dict[fusion_key], f'{fusion_key}_weights_cifar_n10.pth')

# state_dicts_dict = {}

# for key, models_list in party_models_dict.items():
#     state_dicts_dict[key] = [model.state_dict() for model in models_list]

# torch.save(state_dicts_dict, 'party_models_dict_cifar_n10.pth')

################################################################
#load
model_dict = {}
for fusion_key in fusion_types:
   
    model_structure = copy.deepcopy(initial_model)  # 修改为实际模型结构
    model_structure.load_state_dict(torch.load(f'{fusion_key}_weights_cifar_n10.pth',map_location=device))
    model_dict[fusion_key] = model_structure.state_dict()


loaded_state_dicts_dict = torch.load('party_models_dict_cifar_n10.pth',map_location=device)
party_models_dict = {}

for key, state_dicts_list in loaded_state_dicts_dict.items():
    loaded_models_list = []
    for state_dict in state_dicts_list:
        model_instance = copy.deepcopy(initial_model)   # 请替换为你的实际模型结构
        model_instance.load_state_dict(state_dict)
        loaded_models_list.append(model_instance)
    
    party_models_dict[key] = loaded_models_list
###############################################################

for fusion_key in fusion_types:
    current_model_state_dict = model_dict[fusion_key]
    current_model = copy.deepcopy(initial_model)
    current_model.load_state_dict(current_model_state_dict)
    clean_acc = evaluate(testloader_left, testloader_right, current_model)
    print(f'Clean Accuracy {fusion_key}: {clean_acc}')
    pois_acc = evaluate(testloader_poison_left, testloader_poison_right, current_model)
    print(f'Backdoor Accuracy {fusion_key}: {pois_acc}')
    with open("./result_cifar.txt", 'a') as f:
        f.write('\n')
        f.write(f'Clean Accuracy {fusion_key}: {clean_acc}\n')
        f.write(f'Backdoor Accuracy {fusion_key}: {pois_acc}\n') 
 

num_updates_in_epoch = None  
num_local_epochs_unlearn = 50
# lr = 0.01
distance_threshold = 210
# distance_threshold = 2059
clip_grad = 5

fusion_types_unlearn = ['Retrain', 'Unlearn']

# initial_model = SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(device)
unlearned_model_dict = {}
for fusion_key in fusion_types_unlearn:
    if fusion_key == 'Retrain':
        unlearned_model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

clean_accuracy_unlearn = {}
pois_accuracy_unlearn = {}
clean_acc_dict_unlearn = {}
pois_acc_dict_unlearn = {}

for fusion_key in fusion_types_unlearn:
    clean_accuracy_unlearn[fusion_key] = 0
    pois_accuracy_unlearn[fusion_key] = 0
    clean_acc_dict_unlearn[fusion_key] = []
    pois_acc_dict_unlearn[fusion_key] = []

for fusion_key in fusion_types:
    if fusion_key == 'Retrain':
        continue
    starttime = time.process_time()
    # initial_model = SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(device)
    fedavg_model_state_dict = copy.deepcopy(model_dict[fusion_key])
    fedavg_model = copy.deepcopy(initial_model)
    fedavg_model.load_state_dict(fedavg_model_state_dict)

    party_models = copy.deepcopy(party_models_dict[fusion_key])
    party0_model = copy.deepcopy(party_models[0])
    # print(nn.utils.parameters_to_vector(party0_model.parameters()))
    # print(nn.utils.parameters_to_vector(party0_model.clientA.parameters()))
    # print(nn.utils.parameters_to_vector(party0_model.clientB.parameters()))
    # print(nn.utils.parameters_to_vector(party0_model.server.parameters()))
    #compute reference model
    #w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
    alpha = 0.5 
    # model_ref_vec = num_parties / (num_parties - 1) * nn.utils.parameters_to_vector(fedavg_model.parameters()) \
    #                            - 1 / (num_parties - 1) * nn.utils.parameters_to_vector(party0_model.parameters())
    model_ref_vec = (num_parties / ((2*num_parties - 1) * (1 - alpha))) * nn.utils.parameters_to_vector(fedavg_model.parameters()) \
               - (1 - alpha) / ((2*num_parties - 1) * (1 - alpha)) * nn.utils.parameters_to_vector(party0_model.parameters())
    #compute threshold
    model_ref = copy.deepcopy(initial_model)
    # model_ref = copy.deepcopy(fedavg_model)

    nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

    eval_model = copy.deepcopy(model_ref)
    
    unlearn_clean_acc = evaluate(testloader_left, testloader_right, eval_model)
    print(f'Clean Accuracy for Reference Model = {unlearn_clean_acc}')
    unlearn_pois_acc = evaluate(testloader_poison_left, testloader_poison_right, eval_model)
    print(f'Backdoor Accuracy for Reference Model = {unlearn_pois_acc}')
    with open("./result_cifar.txt", 'a') as f:
        f.write('\n')
        f.write(f'Clean Accuracy for Reference Model: {unlearn_clean_acc}\n')
        f.write(f'Backdoor Accuracy for Reference Model: {unlearn_pois_acc}\n') 
    dist_ref_random_lst = []
    for _ in range(10):
        dist_ref_random_lst.append(get_distance(model_ref, SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(device)))    

    print(f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
    threshold = np.mean(dist_ref_random_lst) / 3
    print(f'Radius for model_ref: {threshold}')
    dist_ref_party = get_distance(model_ref, party0_model)
    print(f'Distance of Reference Model to party0_model: {dist_ref_party}')


    ###############################################################
    #### Unlearning
    ###############################################################
    # model = copy.deepcopy(fedavg_model)
    model = copy.deepcopy(model_ref)
    criterion = nn.CrossEntropyLoss()
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    #     
    model.train()
    flag = False
    for epoch in range(num_local_epochs_unlearn):
        print('------------', epoch)
        if flag:
            break
        for (x_batch_left, _),(x_batch_right, y_batch) in zip(trainloader_lst_left_masked[party_to_be_erased], trainloader_lst_right[party_to_be_erased]):
            x_batch_left,x_batch_right, y_batch = x_batch_left.to(device), x_batch_right.to(device), y_batch.to(device)
            model.zero_grad()
            outputs = model(x_batch_left,x_batch_right)
            loss = criterion(outputs, y_batch)
            loss_joint = -loss # negate the loss for gradient ascent
            loss_joint.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            model.step()

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

            # if num_updates_in_epoch is not None and batch_id >= num_updates_in_epoch:
            #     break
    ####################################################################                           
    print(f"\tTime taken: {time.process_time() - starttime}")
    unlearned_model = copy.deepcopy(model)
    unlearned_model_dict[fusion_types_unlearn[1]] = unlearned_model.state_dict() 
    eval_model = SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(device)
    eval_model.load_state_dict(unlearned_model_dict[fusion_types_unlearn[1]])
    unlearn_clean_acc = evaluate(testloader_left, testloader_right, eval_model)
    print(f'Clean Accuracy for UN-Local Model = {unlearn_clean_acc}')
    clean_accuracy_unlearn[fusion_types_unlearn[1]] =  unlearn_clean_acc
    pois_unlearn_acc = evaluate(testloader_poison_left, testloader_poison_right, eval_model)
    print(f'Backdoor Accuracy for UN-Local Model = {pois_unlearn_acc}')
    pois_accuracy_unlearn[fusion_types_unlearn[1]] =  pois_unlearn_acc
    with open("./result_cifar.txt", 'a') as f:
        f.write(f'Clean Accuracy for UN-Local Model: {unlearn_clean_acc}\n')
        f.write(f'Backdoor Accuracy for UN-Local Model: {pois_unlearn_acc}\n') 

    ###########################################



num_fl_after_unlearn_rounds = 1
num_updates_in_epoch = 50
num_local_epochs = 1 

clean_accuracy_unlearn_fl_after_unlearn = {}
pois_accuracy_unlearn_fl_after_unlearn = {}
loss_unlearn = {}
for fusion_key in fusion_types_unlearn:
    clean_accuracy_unlearn_fl_after_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)
    pois_accuracy_unlearn_fl_after_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)
    loss_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)

    
for round_num in range(num_fl_after_unlearn_rounds):

    local_training = LocalTrainingVFL(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

    for fusion_key in fusion_types_unlearn:
        if fusion_key == 'Retrain':
            continue
        # Reduce num_parties by 1 to remove the erased party
        # fusion = FL_round_fusion_selection(num_parties=num_parties - 1, fusion_key=fusion_key)
        fusion = FL_round_fusion_selection(num_parties=num_parties, fusion_key=fusion_key)
        current_model_state_dict = copy.deepcopy(unlearned_model_dict[fusion_key])    
        current_model = copy.deepcopy(initial_model)
        current_model.load_state_dict(current_model_state_dict)

        ##################### Local Training Round #############################
        party_models = []
        party_losses = []
        for party_id in range(0, num_parties):
            model = copy.deepcopy(current_model)
            if party_id == party_to_be_erased:
                model_update, party_loss = local_training.train(splitnn=model, 
                                            trainloaderA=trainloader_lst_left[party_id], 
                                            trainloaderB=trainloader_lst_right_masked[party_id], 
                                            criterion=None)
                party_models.append(copy.deepcopy(model_update))
                party_losses.append(party_loss)

            else:
                model_update, party_loss = local_training.train(splitnn=model, 
                                            trainloaderA=trainloader_lst_left[party_id], 
                                            trainloaderB=trainloader_lst_right[party_id], 
                                            criterion=None)

                party_models.append(copy.deepcopy(model_update))
                party_losses.append(party_loss)

        loss_unlearn[fusion_key][round_num] = np.mean(party_losses)
        ######################################################################

        current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)
        unlearned_model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
        party_models_dict[fusion_key] = party_models  

        eval_model = SplitNN(clientA, clientB, server, optimizerA, optimizerB, optimizerServer).to(device)
        eval_model.load_state_dict(current_model_state_dict)
        unlearn_clean_acc = evaluate(testloader_left, testloader_right, eval_model)
        print(f'Global Clean Accuracy {fusion_key}, round {round_num} = {unlearn_clean_acc}')
        clean_acc_dict_unlearn[fusion_key].append(unlearn_clean_acc)
        clean_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_clean_acc
        unlearn_pois_acc = evaluate(testloader_poison_left, testloader_poison_right, eval_model)
        print(f'Global Backdoor Accuracy {fusion_key}, round {round_num} = {unlearn_pois_acc}')
        pois_acc_dict_unlearn[fusion_key].append(unlearn_pois_acc)
        pois_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_pois_acc

for fusion_key in fusion_types_unlearn:
    if fusion_key == 'Retrain':
            continue
    with open("./result_cifar.txt", 'a') as f:
        f.write('\n')
        f.write(f'Global Clean Accuracy {fusion_key}: {clean_acc_dict_unlearn[fusion_key]} \n')
        f.write(f'Global Backdoor Accuracy {fusion_key}: {pois_acc_dict_unlearn[fusion_key]}\n') 