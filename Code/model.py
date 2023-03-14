import numpy as np
import math
from math import sqrt, pi
import numpy.random as npr
import torch
import itertools


class Network:
    def __init__(self, name='cortex', N=128, S=2, R=2, g=1.5, fb_type='random', seed=1):
        self.name = name
        npr.seed(seed)
        # network parameters
        self.N = N  # RNN units
        self.dt = .1  # time bin (in units of tau)
        self.g = g  # initial recurrent weight scale
        self.S = S  # input
        self.R = R  # readout
        self.sig = 0.001  # initial activity noise
        self.z0 = []    # initial condition
        self.ws = (2 * npr.random((N, S)) - 1) / sqrt(S)  # input weights
        self.J = self.g * npr.standard_normal([N, N]) / np.sqrt(N)  # recurrent weights
        self.wr = (2 * npr.random((R, N)) - 1) / sqrt(N)  # readout weights
        self.fb_type = fb_type
        if fb_type == 'random':
            self.B = npr.standard_normal([N, R]) / sqrt(R)
        elif fb_type == 'aligned':
            self.B = self.wr.T * sqrt(N / R)

    # nlin
    def f(self, x):
        return np.tanh(x) if not torch.is_tensor(x) else torch.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 - np.tanh(x)**2 if not torch.is_tensor(x) else 1 - torch.tanh(x)**2


class Task:
    def __init__(self, name='ComplexSine', duration=20, cycles=2, num_tar=None,
                 rand_init=False, rand_sig=1, dt=0.1):
        self.name = name
        self.rand_init = rand_init
        self.rand_sig = rand_sig
        NT = int(duration / dt)
        # task parameters
        if self.name == 'Reaching':
            self.T, self.dt, self.NT = duration, dt, NT
            self.num_tar = num_tar
            self.s = 0.0 * np.ones((self.num_tar, self.num_tar, NT))
            for cue in range(self.num_tar):
                self.s[cue, cue, 0] = 1  # 1-hot sample
            self.ustar = 0.0 * np.ones((NT, self.num_tar, 2))
        elif self.name == 'ComplexSine':
            self.num_tar = num_tar
            self.cycles, self.T, self.dt, self.NT = cycles, duration, dt, NT
            self.s = 0.0 * np.ones((0, NT))
            self.ustar = (np.sin(2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.75 * np.sin(2 * 2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.5 * np.sin(4 * 2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.25 * np.sin(6 * 2 * pi * np.arange(NT) * cycles / (NT-1)))

    def loss(self, err):
        mse = (err ** 2).mean() / 2
        return mse


class Plant:
    def __init__(self, name='TwoLink'):
        self.name = name
        if self.name == 'TwoLink':
            # physics parameters
            self.noise_scale = 0.2  # network output multiplicative noise scale (0.2)
            self.noise_corr_time = 4  # noise correlation time (units of tau)
            self.drag_coeff = 1.0
            self.noise_feed = (0, 0.2)    # sensory noise (range of SD)
            self.delay_feed = (3, 5)     # sensory delay (units of tau)
            self.w_init = [math.pi / 6, 2 * math.pi / 3]    # initial angles of links
            self.x_init = [0, 1.0]  # initial position of endpoint

    # plant dynamics (actual dynamics with noise)
    def forward(self, u, v, w, noise, dt):
        x = None
        # physics
        if self.name == 'TwoLink':
            accel = u.T + np.linalg.norm(u.T, axis=-1, keepdims=True) * noise - self.drag_coeff * v
            v_new = v + accel * dt
            w = w + v * dt + 0.5 * accel * dt ** 2
            v = v_new
            # hand location
            ang1, ang2 = w[:, 0], w.sum(axis=-1)
            x = np.stack((np.cos(ang1) + np.cos(ang2), np.sin(ang1) + np.sin(ang2)), axis=-1)
        # return
        return v, w, x


class Algorithm:
    def __init__(self, name='rflo', Nepochs=10000, lr=(0, 0, 0, 1e-1, 1e-1), lr_bptt=1e-3, online=True):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.Nstart_anneal = 30000
        self.lr = lr  # learning rate
        self.annealed_lr = 1e-6
        if self.name == 'rflo':
            self.lr_in, self.lr_cc, self.lr_out = lr
            self.online = online
        elif self.name == 'meta':
            self.lr_in, self.lr_cc, self.lr_out = lr
            self.lr_tc = lr_bptt

