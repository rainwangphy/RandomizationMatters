import torch
import numpy as np
# import torch.nn as nn
#
# from advertorch.attacks.base import Attack
# from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import *


def perturb_iterative(xvar, yvar,
                      predictor_list, dis_list,
                      nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                      sparsity=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    :param xvar: input data.
    :param yvar: input labels.

    :param predictor_list: forward pass function.
    :param dis_list: distribution over predictors
    
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param sparsity: the sparsity

    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)
    delta.requires_grad_()
    max_loss_value_iter = -np.inf
    max_adv_iter = torch.zeros_like(xvar)
    for ii in range(nb_iter):
        avg_grad = torch.tensor(xvar.shape).float()
        avg_grad.zero_()
        # if xvar.is_cuda:
        #     avg_grad = avg_grad.cuda()
        # p = []
        loss = torch.tensor(0.)
        # for predict in predictor_list:
        #     p.append(predict.weights / np.sum(predict.weights))
        for i in range(len(predictor_list)):
            outputs = predictor_list[i](xvar + delta)
            loss = loss + loss_fn(outputs, yvar) * dis_list[i]
        if minimize:
            loss = -loss
        loss.backward()

        avg_grad = delta.grad.detach()
        if ord == np.inf:
            grad_sign = avg_grad.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = avg_grad
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            with torch.no_grad():
                grad = avg_grad
                abs_grad = torch.abs(avg_grad)

                batch_size = grad.size(0)
                view = abs_grad.view(batch_size, -1)
                view_size = view.size(1)
                vals, idx = view.topk(int(sparsity * view_size))

                out = torch.zeros_like(view).scatter_(1, idx, vals)

                out = out.view_as(grad)
                grad = grad.sign() * (out > 0).float()
                grad = normalize_by_pnorm(grad, p=1)
                delta.data += batch_multiply(eps_iter, grad)
                delta.data = batch_l1_proj(delta.data.cpu(), eps)
                if xvar.is_cuda:
                    delta.data = delta.data.cuda()
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                   ) - xvar.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)

        delta.grad.data.zero_()

        x_adv = clamp(xvar + delta, clip_min, clip_max)

        loss_2 = torch.tensor(0)
        for i in range(len(predictor_list)):
            outputs_2 = predictor_list[i](xvar + delta)
            loss_2 = loss_2 + loss_fn(outputs_2, yvar) * dis_list[i]

        if max_loss_value_iter < loss_2:
            max_loss_value_iter = loss_2
            max_adv_iter = x_adv

    return max_adv_iter, max_loss_value_iter


class AveragedPGDAttack:

    def __init__(self,
                 estimator_list, distribution_list,
                 loss_function=None,
                 eps=0.3, nb_iter=40,
                 eps_iter=0.01,
                 rand_init=True,
                 clip_min=0., clip_max=1.,
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

    def perturb(self, x, y=None, num_rand_init=3):
        max_loss = -10000
        max_adv_x = torch.zeros_like(x)
        for rand_int in range(num_rand_init):
            delta = torch.zeros_like(x)
            delta = nn.Parameter(delta)
            if self.rand_init:
                rand_init_delta(
                    delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
            adv_x, adv_loss = perturb_iterative(
                x, y,
                predictor_list=self.estimator_list,
                dis_list=self.distribution_list,
                nb_iter=self.nb_iter,
                eps=self.eps, eps_iter=self.eps_iter,
                loss_fn=self.loss_fn, minimize=self.targeted,
                ord=self.ord, clip_min=self.clip_min,
                clip_max=self.clip_max, delta_init=delta, sparsity=self.sparsity)
            if max_loss < adv_loss:
                max_loss = adv_loss
                max_adv_x = adv_x
        return max_adv_x.data
