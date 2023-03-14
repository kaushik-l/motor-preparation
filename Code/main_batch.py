from model import Network, Task, Plant, Algorithm
from train import train_rflo
import torch
import numpy as np
from numpy import random as npr
from matplotlib import pyplot as plt
import sys

modelname = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

if modelname == 'sine_randinit':
    # train using bptt
    lr_in, lr_cc, lr_out = 0, 1e-2, 0
    net, task, algo, learning = \
        train_rflo(arch='m1', N=100, S=0, R=1, task='ComplexSine', duration=10, cycles=1,
                   rand_init=True, rand_sig=(1/10)*np.mod(seed, 11),
                   algo='rflo', fb_type='aligned', Nepochs=10000, lr=(lr_in, lr_cc, lr_out), seed=seed)

# save
torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning},
           '//burg//theory//users//jl5649//motor-preparation//' + modelname + '//' + str(seed) + '.pt')
