import sys
import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from utils import IVRegDataset

def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gain(activation):
    if activation.__class__.__name__ == "LeakyReLU":
        gain = nn.init.calculate_gain("leaky_relu",
                                            activation.negative_slope)
    else:
        activation_name = activation.__class__.__name__.lower()
        try:
            gain = nn.init.calculate_gain(activation_name)
        except ValueError:
            gain = 1.0
    return gain


class MLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, activation=None,last_layer=None, num_out=1):
        nn.Module.__init__(self)
        self.gain=get_gain(activation)
        if len(layer_widths) == 0:
            layers = [nn.Linear(input_dim, num_out)]
        else:
            num_layers = len(layer_widths)
            if activation is None:
                layers = [nn.Linear(input_dim, layer_widths[0])]
            else:
                layers = [nn.Linear(input_dim, layer_widths[0]), activation]
            for i in range(1, num_layers):
                w_in = layer_widths[i-1]
                w_out = layer_widths[i]
                if activation is None:
                    layers.extend([nn.Linear(w_in, w_out)])
                else:
                    layers.extend([nn.Linear(w_in, w_out), activation])
            layers.append(nn.Linear(layer_widths[-1], num_out))
        if last_layer:
            layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def initialize(self, gain=1.0):
        for layer in self.model[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        final_layer = self.model[-1]
        nn.init.xavier_normal_(final_layer.weight.data, gain=gain)
        nn.init.zeros_(final_layer.bias.data)

    def forward(self, data):
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        return self.model(data)

class MultipleMLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, num_models=1, activation=None,last_layer=None, num_out=1):
        nn.Module.__init__(self)
        self.models = nn.ModuleList([MLPModel(
            input_dim, layer_widths, activation=activation,
            last_layer=last_layer, num_out=num_out) for _ in range(num_models)])
        self.num_models = num_models

    def forward(self, data):
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        outputs = [self.models[i](data) for i in range(self.num_models)]
        return torch.cat(outputs, dim=1)

                 
def run(exp, resultDir, train_dict, test_ood_dict, ood_test_dict):
    batch_size = 500
    lr = 0.05
    num_epoch = 5
    mV, mX, mU, mXs = 8, 8, 8, 2
    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/Regression.txt'
    set_seed()

    train = IVRegDataset(data=train_dict)
    test_ood = IVRegDataset(data=test_ood_dict)
    train.to_tensor()
    test_ood.to_tensor()
    train_loader = DataLoader(train, batch_size=batch_size)
    
    input_dim = mV + mX + mXs
    train_input = torch.cat((train.v, train.x),1)
    test_ood_input = torch.cat((test_ood.v, test_ood.x),1)

    
    mlp = MLPModel(input_dim, layer_widths=[128, 64], activation=nn.ReLU(),last_layer=nn.BatchNorm1d(2), num_out=2)
    net = nn.Sequential(mlp)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    log(logfile, f"Exp: {exp}; regt_lr:{lr}")
    for epoch in range(num_epoch):
        log(logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.")
        for idx, inputs in enumerate(train_loader):
            # u = inputs['u']
            v = inputs['v']
            x = inputs['x']
            t = inputs['t'].reshape(-1).type(torch.LongTensor)
            # print("x:", x.shape)
            # print("args.mode:",args.mode)

            input_batch = torch.cat((v, x),1)
            
            prediction = net(input_batch) 
            loss = loss_func(prediction, t)

            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()    

        log(logfile, 'The train accuracy: {:.2f} %'.format((torch.true_divide(sum(train.t.reshape(-1) == torch.max(F.softmax(net(train_input) , dim=1), 1)[1]), len(train.t))).item() * 100))
        log(logfile, 'The test_ood  accuracy: {:.2f} %'.format((torch.true_divide(sum(test_ood.t.reshape(-1) == torch.max(F.softmax(net(test_ood_input) , dim=1), 1)[1]), len(test_ood.t))).item() * 100))

    train_dict['s'] = F.softmax(net(train_input) , dim=1)[:,1:2].cpu().detach().numpy()
    ''' bias rate 1'''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}
    for dt_i in range(len(ood_test_dict)):
        temp = ood_test_dict[dt_i]
        test_ood = IVRegDataset(data=temp)
        try:
            test_ood.to_tensor()
        except:
            pass
        input_dim = mV + mX + mXs
        test_input_ood = torch.cat((test_ood.v, test_ood.x),1)
        ood_test_dict[dt_i]['s'] = F.softmax(net(test_input_ood) , dim=1)[:,1:2].cpu().detach()

    return train_dict, ood_test_dict

# 在第一阶段加入OOD                 
def run_ood_stage1_vx(exp, resultDir, train_dict, test_ood_dict, ood_test_dict):
    batch_size = 500
    lr = 0.05
    num_epoch = 5
    mV, mX, mU, mXs = 2, 8, 8, 2
    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/Regression.txt'
    set_seed()

    train = IVRegDataset(data=train_dict)
    test_ood = IVRegDataset(data=test_ood_dict)
    train.to_tensor()
    test_ood.to_tensor()
    train_loader = DataLoader(train, batch_size=batch_size)
    
    input_dim = mV + mX + mXs
    # print("input dim:", input_dim)
    train_input = train.x
    test_ood_input = test_ood.x

    
    mlp = MLPModel(input_dim, layer_widths=[128, 64], activation=nn.ReLU(),last_layer=nn.BatchNorm1d(2), num_out=2)
    net = nn.Sequential(mlp)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    log(logfile, f"Exp: {exp}; regt_lr:{lr}")
    for epoch in range(num_epoch):
        log(logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.")
        for idx, inputs in enumerate(train_loader):
            # u = inputs['u']
            v = inputs['v']
            x = inputs['x']
            t = inputs['t'].reshape(-1).type(torch.LongTensor)
            # print("x:", x.shape)
            # print("args.mode:",args.mode)

            input_batch = x
            
            prediction = net(input_batch) 
            loss = loss_func(prediction, t)

            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()    

        log(logfile, 'The train accuracy: {:.2f} %'.format((torch.true_divide(sum(train.t.reshape(-1) == torch.max(F.softmax(net(train_input) , dim=1), 1)[1]), len(train.t))).item() * 100))
        log(logfile, 'The test_ood  accuracy: {:.2f} %'.format((torch.true_divide(sum(test_ood.t.reshape(-1) == torch.max(F.softmax(net(test_ood_input) , dim=1), 1)[1]), len(test_ood.t))).item() * 100))

    train_dict['s'] = F.softmax(net(train_input) , dim=1)[:,1:2].cpu().detach().numpy()
    ''' bias rate 1'''
    br = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0]
    brdc = {-3.0: 'n30', -2.5:'n25', -2.0:'n20', -1.5:'n15', -1.3:'n13', 1.3:'p13', 1.5:'p15', 2.0:'p20', 2.5:'p25', 3.0:'p30', 0.0:'0'}
    for dt_i in range(len(ood_test_dict)):
        temp = ood_test_dict[dt_i]
        test_ood = IVRegDataset(data=temp)
        try:
            test_ood.to_tensor()
        except:
            pass
        input_dim = mV + mX + mXs
        test_input_ood = test_ood.x
        ood_test_dict[dt_i]['s'] = F.softmax(net(test_input_ood) , dim=1)[:,1:2].cpu().detach()

    return train_dict, ood_test_dict