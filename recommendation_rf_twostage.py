from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import numpy as np
import torch
from budget import optimize_budget_multilinear, BudgetInstanceMultilinear, dgrad_budget
import pickle
from functools import partial
from optmodel import ContinuousOptimizer
#from multilinear_budget import dgrad_dparams
import torch.nn as nn
import random

#max_depth  = 50
num_iters = 30
opt_vals = {}
kvals = [5, 10, 15, 20, 25, 30]
#kvals = [30]
#kval
#algname = 'dt_{}'.format(max_depth)
algname = 'dt_100'
opt_vals[algname] = np.zeros((num_iters, len(kvals)))
mse_vals = {}
mse_vals[algname] = np.zeros((num_iters))

for idx in range(num_iters):
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
    
    test_pct = 0.2
    num_instances = 500
    
    total_instances = 500
    instances_load = random.sample(range(total_instances), num_instances)
    
    Ps = [load_instance(num_items, i, num_targets) for i in instances_load]
    
    test = random.sample(range(num_instances), int(test_pct*num_instances))
    train = [i for i in range(num_instances) if i not in test]
    
    w = np.ones(num_targets, dtype=np.float32)

    true_transform = nn.Sequential(
            nn.Linear(num_targets, num_targets),
            nn.ReLU(),
            nn.Linear(num_targets, num_targets),
            nn.ReLU(),
            nn.Linear(num_targets, num_targets),
            nn.ReLU(),        
    )
    
    data = [torch.from_numpy(true_transform(P).detach().numpy()).float() for P in Ps]

    loss_fn = nn.MSELoss()    
    
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
    
    Xs = []
    Ys = []
    for i in train:
        Xs.append(data[i].detach().numpy())
        Ys.append(Ps[i].detach().numpy())
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    
    regr = RandomForestRegressor(n_estimators=100, n_jobs=36)
    pickle.dump([1,2,3], open('test.pickle', 'wb'))
    print('start fitting')
    
    regr.fit(X, Y)
    
#    pickle.dump(regr, open('networks/synthetic_dt_100_{}.pickle'.format(idx), 'wb'))
    
    
    def net_two_stage(data_instance):
        y = regr.predict(data_instance.detach().numpy())
        return torch.from_numpy(y).float()
    
    mse_vals[algname][idx] = get_test_mse(net_two_stage)
    
    
    for kidx, k in enumerate(kvals):
        optfunc = partial(optimize_budget_multilinear, w = w, k=k, c = 0.95)
        #dgrad = partial(dgrad_dparams, w=w)
        dgrad = partial(dgrad_budget, w = w)
        opt = ContinuousOptimizer(optfunc, dgrad, None, 0.95)
        opt.verbose = False
        f_true = [BudgetInstanceMultilinear(P, w, True) for P in Ps]
        
        def eval_opt(net, instances):
            val = 0.
            for i in instances:
                pred = net(data[i])
                x = opt(pred)
                val += f_true[i](x)
            return val/len(instances)
        
        opt_vals[algname][idx, kidx] = eval_opt(net_two_stage, test).item()
    print(idx, mse_vals[algname][idx], opt_vals[algname][idx, kidx])
    pickle.dump((opt_vals, mse_vals), open('evaluation_budget_synthetic_{}.pickle'.format(algname), 'wb'))
