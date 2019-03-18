import numpy as np
import torch
from budget import optimize_budget_multilinear, BudgetInstanceMultilinear, dgrad_budget
import pickle
from functools import partial
from optmodel import ContinuousOptimizer
import torch.nn as nn
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--idx', type=int, default=30)

args = parser.parse_args()
num_layers = args.layers
activation = args.activation
idx = args.idx

num_items = 100
two_stage_iters = 2000

cuda = True

with open('movie_data/movie_actor_' + str(num_items) + '.pickle', 'rb') as f:
    Ps, data = pickle.load(f)      

num_instances = len(Ps)
num_targets = Ps[0].shape[1]
num_features = data[0].shape[1]
Ps = [torch.from_numpy(P).float() for P in Ps]
data = [torch.from_numpy(x).float() for x in data]


for idx in range(30):

    with open('movie_data/train_test_splits_' + str(num_items) + '.pickle', 'rb') as f:
        train, test = pickle.load(f)
    
    test = test[idx]
    train = train[idx]    
    
        
    intermediate_size = 200
    
    
    def make_fc():
        if num_layers > 1:
            if activation == 'relu':
                activation_fn = nn.ReLU
            elif activation == 'sigmoid':
                activation_fn = nn.Sigmoid
            else:
                raise Exception('Invalid activation function: ' + str(activation))
#            net_layers = [nn.Linear(num_features, intermediate_size), nn.BatchNorm1d(intermediate_size), nn.Dropout(), activation_fn()]
            net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
            for hidden in range(num_layers-2):
                net_layers.append(nn.Linear(intermediate_size, intermediate_size))
#                net_layers.append(nn.BatchNorm1d(intermediate_size))
#                net_layers.append(nn.Dropout())
                net_layers.append(activation_fn())
            net_layers.append(nn.Linear(intermediate_size, num_targets))
    #        net_layers.append(nn.Sigmoid())
            return nn.Sequential(*net_layers)
        else:
            return nn.Sequential(nn.Linear(num_features, num_targets), nn.Sigmoid())
    
    net = make_fc()
    
    net_two_stage = make_fc()
    if cuda:
        Ps = [P.cuda() for P in Ps]
        data = [d.cuda() for d in data]
        net_two_stage = net_two_stage.cuda()
    
    
    #loss_fn = nn.MultiLabelMarginLoss()
    loss_fn = nn.MultiLabelSoftMarginLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net_two_stage.parameters(), lr=learning_rate, weight_decay=0)
    
    
    def get_test_mse(net):
        loss = 0
        net.eval()
        for i in test:
            pred = net(data[i])
            loss += loss_fn(pred, Ps[i])
        net.train()
        return loss/len(test)
    
    def get_train_mse(net):
        loss = 0
        net.eval()
        for i in train:
            pred = net(data[i])
            loss += loss_fn(pred, Ps[i])
        net.train()
        return loss/len(train)
    
    for t in range(two_stage_iters):
        if t % 100 == 0:
                print(t)
#                total_loss = 0
#                for i in train:
#                    preds = net_two_stage(data[i])
#                    total_loss += loss_fn(preds, Ps[i]).item()
#    #            print(t, total_loss/len(train))
#                
#                total_loss_test = 0
#                for i in test:
#                    preds = net_two_stage(data[i])
#                    total_loss_test += loss_fn(preds, Ps[i]).item()
#                print(t, total_loss/len(train), total_loss_test/len(test))
#                net.train()
    #                torch.cuda.empty_cache()
                print('Train MSE', get_train_mse(net_two_stage).item())
                print('Test MSE', get_test_mse(net_two_stage).item())
    #                print('Test opt', eval_opt(net_two_stage, test).item())
        i = random.choice(train)
        indices = list(range(data[i].shape[0]))
        random.shuffle(indices)
        indices = torch.Tensor(indices).long()
        batch_size = 10
        for batch_idx in range(int(len(indices)/batch_size)):
            minibatch = indices[batch_size*batch_idx:batch_size*(batch_idx+1)]
            pred = net_two_stage(data[i][minibatch])
            loss = loss_fn(pred, Ps[i][minibatch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    net_two_stage.cpu()
    savepath = 'networks/net_soft_moretrain_' + str(num_layers) + '_' + str(idx) + '.pt'
    torch.save(net_two_stage.state_dict(), savepath)