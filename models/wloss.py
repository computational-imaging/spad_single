import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function
import numpy as np
import matplotlib.pyplot as plt

import pdb

dtype = torch.FloatTensor


class WLoss(Function):
    def __init__(self, cost, lam=100, sinkhorn_iter=100):
        super(WLoss, self).__init__()

        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost.type(dtype)
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost / self.lam).type(dtype)
        self.KM = self.cost * self.K
        self.KM = self.KM.type(dtype)
        self.stored_grad = None

        print(self.K)
        print('\n')

    def forward(self, pred, target):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        target = target + torch.ones(target.size()) * 1e-3
        target = target / torch.sum(target)
        # pred = pred + torch.ones(pred.size()).double()*1e-2
        pred = pred + torch.ones(pred.size()) * 1e-3
        pred = pred / torch.sum(pred)

        nbatch = pred.size(0)

        u = self.cost.new(nbatch, self.na).fill_(1.0 / self.na).type(dtype)

        for i in range(self.sinkhorn_iter):
            v = target / (torch.mm(u, self.K.t()))  # double check K vs. K.t() here and next line
            u = pred / (torch.mm(v, self.K))

            if (u != u).sum() > 0 or (
                    v != v).sum() > 0 or u.max() > 1e20 or v.max() > 1e20:  # u!=u is a test for NaN...
                raise Exception(str(
                    ('Warning: numerical errrors', i + 1, "u", (u != u).sum(), u.max(), "v", (v != v).sum(), v.max())))

        loss = (u * torch.mm(v, self.KM.t())).mean(0).sum()  # double check KM vs KM.t()...

        grad = self.lam * u.log() / nbatch
        grad = grad - torch.mean(grad, dim=1).expand_as(grad)

        # plt.plot(grad.data.numpy().transpose())

        self.stored_grad = grad

        dist = self.cost.new((loss,))
        return dist

    def backward(self, grad_output):
        return self.stored_grad * grad_output[0], None