import random
import glob
import json
import time

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from SimpleDataLoader import CustomDataset, get_params_from_filename
import numpy as np
from DNN_model import Net
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from MMS_compute import xpress_solver
import copy


path_to_data = 'Dataset'

def split_to_train_validation(path_to_data):

    dataset = CustomDataset(path_to_data)
    print(len(dataset))

    batch_size = 300
    validation_split = 0.2
    shuffle_dataset = True
    random_seed= 56
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print(len(train_indices), len(val_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    print(len(train_loader), len(validation_loader))
    return train_loader, validation_loader


train_loader, validation_loader = split_to_train_validation(path_to_data)

net = Net()





loss_func = nn.MSELoss()
# loss_func = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)


def compute_loss(dataloader, net):
    loss = 0

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    n_batches = 0
    with torch.no_grad():
        for x, y in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            pred = net(x)

            loss += loss_func(pred, y).item()

    loss = loss / n_batches
    return loss




n_epochs = 50

pbar = tqdm(range(n_epochs))
validation_loss_vs_epoch = []

if torch.cuda.is_available():
    net.cuda()

for epoch in pbar:

    if len(validation_loss_vs_epoch) > 1:
        print('epoch', epoch, ' val loss:' + '{0:.5f}'.format(validation_loss_vs_epoch[-1]))

    net.train()  # put the net into "training mode"
    for x, y in train_loader:
        y = y.to(torch.float32)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        pred = net(x)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode

    valid_loss = compute_loss(validation_loader, net)

    validation_loss_vs_epoch.append(valid_loss)

# n = 5
# m = 50
# max_val = 100
# values = [random.randrange(0, max_val + 1) for _ in range(m)]
# values.sort(reverse=True)
# values += [0]*50
# mms = xpress_solver(values,n)[0]
# sum_vals = sum(values)
# new_values = [val/sum_vals for val in values]
# pred = net(torch.FloatTensor([float(n)]+new_values))
# pred_num = float(pred.data[0])
# print(pred, mms, pred*sum_vals)
# print(pred_num*sum_vals)


def zero_pad(values, max_m):
    m = len(values)
    values += [0] * (max_m - m)


def solve_with_solver(values_copy, n):
    return xpress_solver(values_copy, n)



def solve_with_net(values_copy, n):
    start = time.time()
    sum_vals = sum(values_copy)
    new_values = [val / sum_vals for val in values_copy]
    pred = net(torch.FloatTensor([float(n)] + new_values))
    pred_num = float(pred.data[0])
    final_result =  pred_num*sum_vals
    end = time.time()
    return final_result, end-start

def test_net(path):
    max_m = 100
    filelist = glob.glob(path + '/*.json')
    print(len(filelist))

    test_result = dict()
    filelist_len = len(filelist)
    for count, filename in enumerate(filelist):
        n, m, max_val = get_params_from_filename(filename)
        data_list_in_file = []
        with open(filename) as jsonFile:
            data_list_in_file = json.load(jsonFile)
        idx = random.randint(0, len(data_list_in_file)-1)
        example=data_list_in_file[idx]
        values = example[0]["values"]
        values_copy = copy.deepcopy(values)
        values_copy.sort(reverse=True)
        solver_result, solver_time = solve_with_solver(values_copy, n)

        zero_pad(values_copy, max_m)
        net_result, net_time = solve_with_net(values_copy, n)
        test_result[str((n, m, max_val))] = {
            'values_idx': idx,
            'solver_result': solver_result,
            'solver_time':solver_time,
            'net_result':net_result,
            'net_time':net_time
        }
        if count % 20 == 0:
            print(count, 'out of', filelist_len)
    test_result_path = './TestResults/test_results.json'
    with open(test_result_path, 'w+') as json_file:
        json.dump(test_result, json_file, indent=4)

test_net(path_to_data)