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
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--k', type=int, default=20)

args = parser.parse_args()
num_layers = args.layers
activation = args.activation
k = args.k
use_hessian = False
num_iters = 400
instance_sizes = [100]
learning_rate = 1e-4

Ps = {}
data = {}
f_true = {}
for num_items in instance_sizes:
    with open('movie_data/movie_actor_' + str(num_items) + '.pickle', 'rb') as f:
        Ps_size, data_size = pickle.load(f)      
    
    num_targets = Ps_size[0].shape[1]
    num_features = data_size[0].shape[1]
    Ps[num_items] = [torch.from_numpy(P).long() for P in Ps_size]
    data[num_items] = [torch.from_numpy(x).float() for x in data_size]
    w = np.ones(num_targets, dtype=np.float32)
    f_true[num_items] = [CoverageInstanceMultilinear(P, w, True) for P in Ps[num_items]]
    

num_repetitions = 0

train = {}
test = {}
for size in instance_sizes:
    with open('movie_data/train_test_splits_' + str(size) + '.pickle', 'rb') as f:
        train[size], test[size] = pickle.load(f)


vals = np.zeros((num_repetitions+30, len(instance_sizes), len(instance_sizes)))

for idx in range(num_repetitions, num_repetitions + 30):
    
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
            net_layers.append(nn.Sigmoid())
            return nn.Sequential(*net_layers)
        else:
            return nn.Sequential(nn.Linear(num_features, num_targets), nn.Sigmoid())
    
    
    #optimizer that will be used for training (and testing)
    optfunc = partial(optimize_coverage_multilinear, w = w, k=k, c = 0.95)
    dgrad = partial(dgrad_coverage, w = w)
    if use_hessian:
        hessian = partial(hessian_coverage, w = w)
    else:
        hessian = None
    opt = ContinuousOptimizer(optfunc, dgrad, None, 0.95)
    opt.verbose = False

    #runs the given net on instances of a given size
    def eval_opt(net, instances, size):
        net.eval()
        val = 0.
        for i in instances:
            pred = net(data[size][i])
            x = opt(pred)
            val += f_true[size][i](x)
        net.train()
        return val/len(instances)
    
    #train a network for each size, and test on each sizes
    for train_idx, train_size in enumerate(instance_sizes):
        net = make_fc()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        #training
        for t in range(num_iters):
            print(t)
            i = random.choice(train[train_size][idx])
            pred = net(data[train_size][i])
            x = opt(pred)
            loss = -f_true[train_size][i](x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #save learned network state
        savepath = 'networks/net_diffopt_smalllr_{0}_{1}_{2}_{3}.pt'.format(train_size, k, num_layers, idx)
        torch.save(net.state_dict(), savepath)
        #test on different sizes
        for test_idx, test_size in enumerate(instance_sizes):
            vals[idx, train_idx, test_idx] = eval_opt(net, test[test_size][idx], test_size)
            print(vals[idx, train_idx, test_idx])
        #save out values
        print(idx, train_size, vals[idx, train_idx])
        with open('results_recommendation_' + str(num_layers) + '.pickle', 'wb') as f:
            pickle.dump(vals, f)