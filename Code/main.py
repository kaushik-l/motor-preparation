from model import Network, Task, Plant, Algorithm
from train import train_rflo
import torch
import numpy as np
from numpy import random as npr
from matplotlib import pyplot as plt

rflo_sine = False
rflo_reach = True

if rflo_sine:
    lr_in, lr_cc, lr_out = 0, 1e-2, 0
    net, task, algo, learning = train_rflo(arch='m1', N=100, S=0, R=1, task='ComplexSine', duration=20, cycles=2,
                                           rand_init=True, rand_sig=0.5,
                                           algo='rflo', fb_type='aligned', Nepochs=10000, lr=(lr_in, lr_cc, lr_out), seed=1)
    # save
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, '..\\Data\\sine_randinit.pt')

if rflo_reach:
    lr_in, lr_cc, lr_out = 0, 1e-2, 1e-2
    net, task, algo, learning = train_rflo(arch='m1', N=100, S=4, R=2, task='Reaching', duration=20, cycles=2,
                                           rand_init=False, rand_sig=0.5, num_tar=4,
                                           algo='rflo', fb_type='aligned', Nepochs=10000, lr=(lr_in, lr_cc, lr_out), seed=1)
    # save
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, '..\\Data\\reach_randinit.pt')

##
# data = torch.load('..//Data//sine_randinit.pt')
# indx = np.argsort([data['task'][k,0].rand_sig for k in range(550)])
# plt.errorbar(range(11),
#              np.reshape([data['learning'][k,0]['mses_test'].squeeze().mean() for k in indx], (11,50)).mean(axis=1),
#              np.reshape([data['learning'][k,0]['mses_test'].squeeze().mean() for k in indx], (11,50)).std(axis=1)/7)