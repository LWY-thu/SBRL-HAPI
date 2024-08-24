'''
Author: your name
Date: 2024-08-24 15:22:37
LastEditTime: 2024-08-24 15:22:37
LastEditors: your name
Description: In User Settings Edit
FilePath: /liuwenyang05/code/SBRL-HAPI/uitls/loadDataset.py
'''
'''
Author: your name
Date: 2024-05-28 16:07:36
LastEditTime: 2024-05-30 15:27:22
LastEditors: ai-platform-wlf1-ge2-127.idchb2az1.hb2.kwaidc.com
Description: In User Settings Edit
FilePath: /liuwenyang03/code/Revision-v1-2-anpeng-cfr-hsic-py2.7/loadDataset.py
'''
from torch.utils.data import Dataset
import torch
import numpy as np

class IVRegDataset(Dataset):
    """
    Syn Dataset
    """
    def __init__(self, data):
        self.x = data['x']
        self.v = data['v']
        self.t = data['t']
        self.yf = data['yf']
        self.ycf = data['ycf']
        self.mu0 = data['mu0']
        self.mu1 = data['mu1']
        self.t_ = data['t'].astype(np.bool)
        self.variables = ['x', 'v','t','yf','ycf','mu0','mu1','t_']


    def __getitem__(self, index):
        var_dict = {}
        for var in self.variables:
            exec(f'var_dict[\'{var}\']=self.{var}[index]')
        
        return var_dict

    def add_var(self, var):
        exec(f'self.{var} = None')
        exec(f'self.variables.append(\'{var}\')')

    def __len__(self):
        return self.t.shape[0]

    def to_cpu(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cpu()')
            
    def to_cuda(self,n=0):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cuda({n})')
    
    def to_tensor(self):
        for var in self.variables:
            exec(f'self.{var} = torch.Tensor(self.{var})')
            
    def to_double(self):
        for var in self.variables:
            exec(f'self.{var} = torch.Tensor(self.{var}).double()')
            
    def to_numpy(self):
        try:
            self.detach()
            self.to_cpu()
        except:
            self.to_cpu()
        for var in self.variables:
            exec(f'self.{var} = self.{var}.numpy()')
    
    def detach(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.detach()')