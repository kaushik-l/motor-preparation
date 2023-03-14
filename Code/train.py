import torch
import numpy as np
import numpy.random as npr
from model import Network, Task, Algorithm, Plant
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.linalg import subspace_angles
from matplotlib import pyplot as plt
import copy

# seed
npr.seed(3)


def unstack(a, axis=0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]


def dimensionality(x):
    return (x.sum() ** 2) / (x ** 2).sum()


def simulate(net, task):
    # frequently used vars
    S, N, R, NT = net.S, net.N, net.R, task.NT
    # initialize output
    ha = np.zeros((NT, N))  # cortex hidden states
    ua = np.zeros((NT, R))  # angular acceleration of joints
    err = np.zeros((NT, R))  # error
    # initialize activity
    z0 = net.z0 + task.rand_sig*npr.randn(N, 1) if task.rand_init else net.z0  # hidden state (potential)
    h0 = net.f(z0)  # hidden state (rate)
    z, h = z0, h0
    for ti in range(NT):
        # currents
        s = task.s[:, ti]
        Iin = np.matmul(net.ws, s)[:, None] if net.S else 0
        Irec = np.matmul(net.J, h)
        z = Iin + Irec  # potential
        # update activity
        h = (1 - net.dt) * h + net.dt * (net.f(z))  # cortex
        u = np.matmul(net.wr, h)  # output
        # error
        err[ti] = task.ustar[ti] - u
        # save values
        ha[ti], ua[ti] = h.T, u.T
    mse = task.loss(err)
    return ha, ua, mse


# train using rflo
def train_rflo(arch='ctx', N=256, S=2, R=2, g=1.5, task='ComplexSine', duration=20, cycles=2,
               rand_init=False, rand_sig=1, num_tar=0,
               algo='rflo', fb_type='random', Nepochs=10000, lr=(0, 1e-1, 1e-1), online=True,
               Nepochs_test=100, seed=1):

    # instantiate model
    net = Network(arch, N, S, R, g=g, fb_type=fb_type, seed=seed)
    task = Task(task, duration, cycles, rand_init=rand_init, rand_sig=rand_sig, num_tar=num_tar)
    algo = Algorithm(algo, Nepochs, lr, online=online)

    # frequently used vars
    dt, NT, N, S, R = net.dt, task.NT, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # track variables during learning
    learning = {
        'epoch': [], 'lr': [], 'mses': [], 'mses_test': [], 'ua_test': [], 'J0': np.empty_like(net.J),
        'alignment': {'feedback': []}
    }

    # random initialization of hidden state
    z0 = npr.randn(N, 1)    # hidden state (potential)
    net.z0 = z0  # save

    # save initial weights
    learning['J0'][:] = net.J

    for ei in range(algo.Nepochs):

        # initialize activity
        z0 = net.z0 + task.rand_sig*npr.randn(N, 1) if task.rand_init else net.z0  # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = z0, h0

        # save tensors for plotting
        sa = np.zeros((NT, S))  # save the inputs for each time bin for plotting
        ha = np.zeros((NT, N))  # save the hidden states for each time bin for plotting
        ua = np.zeros((NT, R))  # angular acceleration of joints

        # errors
        err = np.zeros((NT, R))     # error in angular acceleration

        # eligibility trace q
        q = net.df(z) * ha[0]

        # store weight changes for offline learning
        dws = np.zeros_like(net.ws)
        dwr = np.zeros_like(net.wr)
        dJ = np.zeros_like(net.J)

        for ti in range(NT):

            # network update
            s = task.s[:, ti]
            Iin = np.matmul(net.ws, s)[:, None] if net.S else 0
            Irec = np.matmul(net.J, h)
            z = Iin + Irec    # potential

            # update eligibility trace
            if algo.lr_in:
                o = dt * net.df(z) * s.T + (1 - dt) * o
            if algo.lr_cc:
                q = dt * net.df(z) * h.T + (1 - dt) * q

            # update activity
            h = (1 - dt) * h + dt * (net.f(z))  # cortex
            u = np.matmul(net.wr, h)  # output

            # save values for plotting
            sa[ti], ha[ti], ua[ti] = s.T, h.T, u.T

            # error
            err[ti] = task.ustar[ti] - u

            # online weight update
            if algo.online:
                if algo.lr_in:
                    net.ws += ((algo.lr_in / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * o)
                if algo.lr_out:
                    net.wr += (((algo.lr_out / NT) * h) * err[ti]).T
                if algo.lr_cc:
                    net.J += ((algo.lr_cc / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * q)
                net.B = net.wr.T * np.sqrt(N / R) if fb_type == 'aligned' else net.B    # realign feedback if needed
            else:
                if algo.lr_in:
                    dws += ((algo.lr_in / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * o)
                if algo.lr_out:
                    dwr += (((algo.lr_out / NT) * h) * err[ti]).T
                if algo.lr_cc:
                    dJ += ((algo.lr_cc / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * q)

        # offline update
        if not algo.online:
            net.ws += dws
            net.wr += dwr
            net.J += dJ
            net.B = net.wr.T * np.sqrt(N / R) if fb_type == 'aligned' else net.B    # realign feedback if needed

        # compute overlap
        learning['alignment']['feedback'].append((net.wr.flatten() @ net.B.flatten('F')) /
                                                 (np.linalg.norm(net.wr.flatten()) * np.linalg.norm(net.B.flatten('F'))))

        # print loss
        mse = task.loss(err)
        if (ei+1) % 10 == 0:
            print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(mse), end='')

        # save mse list and cond list
        learning['mses'].append(mse)

        # adaptive learning rate
        if algo.lr_cc:
            algo.lr_cc *= np.exp(np.log(np.minimum(1e-1, algo.lr_cc) / algo.lr_cc) / Nepochs)
            learning['lr'].append(algo.lr_cc)
            learning['epoch'].append(ei)

    # simulate 100 trials
    ua_test = np.zeros((Nepochs_test, NT, R))
    mse_test = np.zeros((Nepochs_test, 1))
    for idx in np.arange(Nepochs_test):
        _, ua_test[idx], mse_test[idx] = simulate(net, task)
    learning['ua_test'], learning['mses_test'] = ua_test[:10], mse_test

    return net, task, algo, learning
