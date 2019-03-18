import torch
import torch.nn as nn
import random
import numpy as np
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=2)

args = parser.parse_args()
num_layers = args.layers
print(num_layers)


def make_matching_matrix(n):
    
    lhs = list(range(n))
    rhs = list(range(n, 2*n))
    
    n_vars = len(lhs)*len(rhs)
    n_constraints = len(lhs) + len(rhs) + n_vars
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros((n_constraints))
    curr_idx = 0
    edge_idx = {}
    for u in lhs:
        for v in rhs:
            edge_idx[(u,v)] = curr_idx
            curr_idx += 1
    for u in lhs:
        for v in rhs: 
            A[u, edge_idx[(u,v)]] = 1
            A[v, edge_idx[(u,v)]] = 1
            A[len(lhs)+len(rhs)+edge_idx[(u,v)], edge_idx[(u,v)]] = -1
            
    for u in lhs:
        b[u] = 1
    for u in rhs:
        b[u] = 1
    
    return A, b
    
            
                  
load = True
if load:
    Ps = torch.load('../cora_graphs_bipartite.pt')
    data = torch.load('../cora_features_bipartite.pt')
    Ps = Ps.view(*Ps.shape, 1)

activation = 'relu'
intermediate_size=500
num_features = data.shape[2]
num_targets = 100
cuda = False

n_instances = data.shape[0]

algs = ['diffopt', 'ce', 'random']
num_iters = 30
auc = {}
ce_loss = {}
opt = {}
optimum = []
for alg in algs:
    auc[alg] = np.zeros((num_iters))
    ce_loss[alg] = np.zeros((num_iters))
    opt[alg] = np.zeros((num_iters))

for iter_idx in range(num_iters):
    test = random.sample(list(range(n_instances)), int(0.2*n_instances))
    train = [i for i in range(n_instances) if i not in test]
    two_stage_iters = len(train)
    #two_stage_iters=1
    
        
    def make_fc(num_layers, num_features, num_targets, regularizers = False):
        if num_layers > 1:
            if activation == 'relu':
                activation_fn = nn.ReLU
            elif activation == 'sigmoid':
                activation_fn = nn.Sigmoid
            else:
                raise Exception('Invalid activation function: ' + str(activation))
            if regularizers:
                net_layers = [nn.Linear(num_features, intermediate_size), nn.Dropout(), activation_fn()]
            else:
                net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
            for hidden in range(num_layers-2):
                net_layers.append(nn.Linear(intermediate_size, intermediate_size))
                if regularizers:
    #                net_layers.append(nn.BatchNorm1d(intermediate_size))
                    net_layers.append(nn.Dropout())
                net_layers.append(activation_fn())
            net_layers.append(nn.Linear(intermediate_size, num_targets))
    #        net_layers.append(nn.Sigmoid())
            return nn.Sequential(*net_layers)
        else:
            return nn.Sequential(nn.Linear(num_features, num_targets), nn.Sigmoid())
        
    net = make_fc(num_layers, num_features = num_features, num_targets = 1, regularizers=False)
    
    net_two_stage = make_fc(num_layers, num_features = num_features, num_targets = 1, regularizers=False)
    cuda = False
    if cuda:
    #    Ps = [P.cuda() for P in Ps]
    #    data = [d.cuda() for d in data]
        net_two_stage = net_two_stage.cuda()
    
    
    #loss_fn = nn.MultiLabelMarginLoss()
    loss_fn = nn.BCEWithLogitsLoss()
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
    
    #for epoch in range(two_stage_iters):
    #    print(epoch)
        
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
    #                print('Test opt', eval_opt(net_two_stage, test).item())
    #    i = random.choice(train)
    verbose = False
    batch_size = 100
    for epoch in range(2):
        random.shuffle(train)
        for idx, i in enumerate(train[:two_stage_iters]):
#            print(idx)
            if verbose and idx % 10 == 0:
                net.eval()
                print('Train MSE', get_train_mse(net_two_stage).item())
                print('Test MSE', get_test_mse(net_two_stage).item())
                net.train()
            
            if cuda:
                X = data[i].cuda()
                Y = Ps[i].cuda
            else:
                X = data[i]
                Y = Ps[i]
        
            n_samples = X.shape[0]
            order = torch.randperm(n_samples)
            for t in range(int(n_samples/batch_size)):
                minibatch = order[t*batch_size:(t+1)*batch_size]
                pred = net_two_stage(X[minibatch])
                loss = loss_fn(pred, Y[minibatch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    import sklearn.metrics 
    
    def get_auc(net):
        net.eval()
        aucvals = []
        for i in test:
            y_true = Ps[i].detach().numpy().flatten()
            y_score = nn.Sigmoid()(net(data[i])).detach().numpy().flatten()
            auc_i = sklearn.metrics.roc_auc_score(y_true, y_score)
    #        print(auc_i)
            aucvals.append(auc_i)
        net.train()
        return np.mean(aucvals)
    
    auc['ce'][iter_idx] = get_auc(net_two_stage)
#    print(auc['ce'][iter_idx])
#    print('twostage auc: {0}'.format(auc['ce'][iter_idx]))
    
    #loss_ts = get_loss(net_two_stage, data[test], Ps[test], model_params_linear, torch.zeros(A.shape[1], A.shape[1]), A, b)
    #print(loss_ts)
    
#    raise Exception()
        
    A, b = make_matching_matrix(50)
    A = torch.from_numpy(A).float()
    b = torch.from_numpy(b).float()
    
    def get_loss(net, data, c_true, model_params, Q, G, h, eval_mode = True):
        if eval_mode:
            net.eval()
        c_pred = -nn.Sigmoid()(net(data))
        if c_pred.dim() == 3:
            n_train = data.shape[0]
        else:
            n_train = 1
        c_pred = c_pred.squeeze()
    #    print(n_train)
    #    x_1 = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
    #    x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
        x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
    #    return x
        loss = (c_true.view(c_true.shape[0], 1, c_true.shape[1])@x.view(*x.shape, 1)).mean()
        net.train()
        return loss
    
    def get_loss_random(data, c_true, model_params, Q, G, h):
        c_pred = -torch.rand_like(c_true)
        if c_pred.dim() == 3:
            n_train = data.shape[0]
        else:
            n_train = 1
        c_pred = c_pred.squeeze()
    #    print(n_train)
    #    x_1 = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
    #    x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
        x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
    #    return x
        loss = (c_true.view(c_true.shape[0], 1, c_true.shape[1])@x.view(*x.shape, 1)).mean()
        return loss
    
    def get_loss_opt(data, c_true, model_params, Q, G, h):
#        c_pred = -torch.rand_like(c_true)
        c_pred = -c_true
        if c_pred.dim() == 3:
            n_train = data.shape[0]
        else:
            n_train = 1
        c_pred = c_pred.squeeze()
    #    print(n_train)
    #    x_1 = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
    #    x = QPFunction(verbose=False, solver=QPSolvers.GUROBI)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
        x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, model_params=model_params)(Q.expand(n_train, *Q.shape), c_pred, G.expand(n_train, *G.shape), h.expand(n_train, *h.shape), torch.Tensor(), torch.Tensor())
    #    return x
#        print(x)
#        print(x.sum())
        loss = (c_true.view(c_true.shape[0], 1, c_true.shape[1])@x.view(*x.shape, 1)).mean()
        return loss

    
    gamma = 0.1
    #loss_ts = get_loss(net_two_stage, data[[0]], Ps[[0]], torch.zeros(A.shape[1], A.shape[1]), A, b)
    #qp_linear = QPFunction(verbose=False, solver=QPSolvers.GUROBI, constant_constraints=True, G=A, h=b, Q=torch.zeros(A.shape[1], A.shape[1]))
    #qp_quad = QPFunction(verbose=False, solver=QPSolvers.GUROBI, constant_constraints=True, G=A, h=b, Q=gamma*torch.eye(A.shape[1]))
    
    
    model_params_linear = make_gurobi_model(A.detach().numpy(), b.detach().numpy(), None, None, np.zeros((A.shape[1], A.shape[1])))
    model_params_quad = make_gurobi_model(A.detach().numpy(), b.detach().numpy(), None, None, gamma*np.eye(A.shape[1]))
    
    loss_opt = get_loss_opt(data[test], Ps[test], model_params_linear, torch.zeros(A.shape[1], A.shape[1]), A, b)
    print(loss_opt.item())
    optimum.append(loss_opt.item())
#    continue
    
    loss_ts = get_loss(net_two_stage, data[test], Ps[test], model_params_linear, torch.zeros(A.shape[1], A.shape[1]), A, b)
    #print('two stage', loss_ts)
    loss_random = get_loss_random(data[test], Ps[test], model_params_linear, torch.zeros(A.shape[1], A.shape[1]), A, b)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    for epoch in range(12):
        print(epoch)
        random.shuffle(train)
        for i in train:
            loss = -get_loss(net, data[[i]], Ps[[i]], model_params_quad, gamma*torch.eye(A.shape[1]), A, b, eval_mode=False)
#            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        gamma *= 0.8
        print(gamma)
    loss_diffopt = get_loss(net, data[test], Ps[test], model_params_linear, torch.zeros(A.shape[1], A.shape[1]), A, b)
    print(epoch, 'test', loss_diffopt)
    
    print(iter_idx)
    opt['ce'][iter_idx] = loss_ts.item()
    opt['diffopt'][iter_idx] = loss_diffopt.item()
    auc['diffopt'][iter_idx] = get_auc(net)
    opt['random'][iter_idx] = loss_random.item()
    ce_loss['diffopt'][iter_idx] = get_test_mse(net).item()
    ce_loss['ce'][iter_idx] = get_test_mse(net_two_stage).item()
    
    print('OPT: {0} {1} {2}'.format(opt['diffopt'][iter_idx], opt['ce'][iter_idx], opt['random'][iter_idx]))
    print('AUC: {0} {1}'.format(auc['diffopt'][iter_idx], auc['ce'][iter_idx]))
    print('CE: {0} {1}'.format(ce_loss['diffopt'][iter_idx], ce_loss['ce'][iter_idx]))
    
#    torch.save(net.state_dict(), 'cora_diffopt_1.pt')
#    torch.save(net_two_stage.state_dict(), 'cora_ts_1.pt')
    
    pickle.dump((opt, auc, ce_loss), open('results_cora_{}_cameraready.pickle'.format(num_layers), 'wb'))
#print('average optimum', np.mean(optimum))