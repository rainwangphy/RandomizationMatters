import torch
import numpy as np
import torch.nn as nn

from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import *


def perturb_iterative(xvar, yvar, estimator_list, distribution_list, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0, sparsity=0.01):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.

    :return: tensor containing the perturbed input.
    """
    max_loss_value = -100000
    return 0


class AveragedPGDAttack():

    def __init__(self, estimator_list, distribution_list,
                 loss_function=None,
                 eps=0.3, nb_iter=40,
                 eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                 ord=np.inf, targeted=False, sparsity=0.01):

        self.estimator_list = estimator_list
        self.distribution_list = distribution_list
        self.loss_function = loss_function

        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted

        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.loss_function is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.sparsity = sparsity

    def perturb(self, x, y=None):
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        rval = perturb_iterative(
            x, y,
            estimator_list=self.estimator_list, distribution_list=self.distribution_list,
            nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta, sparsity=self.sparsity)

        return rval.data
