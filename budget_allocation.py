import numpy as np
import torch
from coverage import optimize_coverage_multilinear, CoverageInstanceMultilinear, dgrad_coverage, hessian_coverage
import pickle
from functools import partial
from submodular import ContinuousOptimizer
import torch.nn as nn
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=2)

args = parser.parse_args()
num_layers = args.layers

def load_instance(n, i, num_targets):
    with open('new_budget_instances/yahoo_' + str(n) + '_' + str(i), 'rb') as f:
        Pfull, wfull = pickle.load(f, encoding='bytes')
    P = np.zeros((num_items, num_targets), dtype=np.float32)
    for i in range(num_targets):
        for j in Pfull[i]:
            P[j, i] = Pfull[i][j]
    P = torch.from_numpy(P).float()
    return P
    

num_items = 100
num_targets = 500
num_iters = 40
use_hessian = True
test_pct = 0.2
num_instances = 1000

total_instances = 1000
instances_load = random.sample(range(total_instances), num_instances)

Ps = [load_instance(num_items, i, num_targets) for i in instances_load]


kvals = [5, 10, 20]

opt_vals = {}
mse_vals = {}
algs = ['diffopt', 'twostage', 'opt', 'random']

for alg in algs:
    opt_vals[alg] = np.zeros((30, len(kvals)))
    mse_vals[alg] = np.zeros((30, len(kvals)))


activation = 'relu'
intermediate_size = 200

def make_fc():
    if num_layers > 1:
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
        for hidden in range(num_layers-2):
            net_layers.append(nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        net_layers.append(nn.Linear(intermediate_size, num_targets))
        net_layers.append(nn.ReLU())
        return nn.Sequential(*net_layers)
    else:
        return nn.Sequential(nn.Linear(num_features, num_targets), nn.ReLU())


idx = 0
for idx in range(30):
    print(idx)
    test = random.sample(range(num_instances), int(test_pct*num_instances))
    train = [i for i in range(num_instances) if i not in test]
    
    
    w = np.ones(num_targets, dtype=np.float32)
    num_features = int(num_targets)
    true_transform = nn.Sequential(
            nn.Linear(num_targets, num_targets),
            nn.ReLU(),
            nn.Linear(num_targets, num_targets),
            nn.ReLU(),
            nn.Linear(num_targets, num_features),
            nn.ReLU(),    
    
    )
    
    data = [torch.from_numpy(true_transform(P).detach().numpy()).float() for P in Ps]
    f_true = [CoverageInstanceMultilinear(P, w, True) for P in Ps]
    
    net = make_fc()
    net_two_stage = make_fc()
        
    
    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net_two_stage.parameters(), lr=learning_rate)
    
    
    def get_test_mse(net):
        loss = 0
        for i in test:
            pred = net(data[i])
            loss += loss_fn(pred, Ps[i])
        return loss/len(test)
    
    def get_train_mse(net):
        loss = 0
        for i in train:
            pred = net(data[i])
            loss += loss_fn(pred, Ps[i])
        return loss/len(train)
    
    
    def get_test_mse_random():
        loss = 0
        train_sum = 0
        for i in train:
            train_sum += Ps[0].sum()
        train_sum /= len(train)
        for i in test:
            pred = torch.rand(num_items, num_targets).float()
            pred *= train_sum/pred.sum()
            loss += loss_fn(pred, Ps[i])
        return loss/len(test)
    
    
    print('train two stage')
    for t in range(4001):
        i = random.choice(train)
        pred = net_two_stage(data[i])
        loss = loss_fn(pred, Ps[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mse_vals['twostage'][idx, 0] = get_test_mse(net_two_stage).item()
            
    for kidx, k in enumerate(kvals):
        optfunc = partial(optimize_coverage_multilinear, w = w, k=k, c = 0.95)
        dgrad = partial(dgrad_coverage, w = w)
        if use_hessian:
            hessian = partial(hessian_coverage, w = w)
        else:
            hessian = None
        opt = ContinuousOptimizer(optfunc, dgrad, hessian, 0.95)
        opt.verbose = False
        
        
        def eval_opt(net, instances):
            val = 0.
            for i in instances:
                pred = net(data[i])
                x = opt(pred)
                val += f_true[i](x)
            return val/len(instances)
        
        def get_opt(instances):
            val = 0.
            for i in instances:
                x = opt(Ps[i])
                val += f_true[i](x)
            return val/len(instances)
        
        def get_rand(instances):
            val = 0
            for _ in range(100):
                for i in instances:
                    x = np.zeros(num_items)
                    x[random.sample(range(num_items), k)] = 1
                    x = torch.from_numpy(x).float()
                    val += f_true[i](x)
            return val/(100*len(instances))
        
        
        opt_vals['opt'][idx, kidx] = get_opt(test).item()
        opt_vals['twostage'][idx, kidx] = eval_opt(net_two_stage, test).item()
        opt_vals['random'][idx, kidx] = get_rand(test).item() 
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for t in range(num_iters):
            i = random.choice(train)
            pred = net(data[i])
            x = opt(pred)
            loss = -f_true[i](x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        opt_vals['diffopt'][idx, kidx] = eval_opt(net, test).item()
        mse_vals['diffopt'][idx, kidx] = get_test_mse(net).item()
        for alg in algs:
            print(alg, opt_vals[alg][idx, kidx])
    pickle.dump(opt_vals, open('evaluation_synthetic_full_{}_opt.pickle'.format(num_layers), 'wb'))
    pickle.dump(mse_vals, open('evaluation_synthetic_full_{}_mse.pickle'.format(num_layers), 'wb'))
    