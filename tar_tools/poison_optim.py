import torch
import numpy as np


def pgd(inputs, labels, model, loss_fn, dm, ds, eps=8, steps=10, alpha=2, rand_init=True):
    model.eval()

    if rand_init:
        x_adv = inputs.clone().detach() + torch.from_numpy(np.random.uniform(-eps / 255., eps / 255., inputs.shape)).float().cuda()

    eps = eps / ds / 255.
    alpha = alpha / ds / 255.

    for _ in range(steps):
        x_adv.requires_grad_()
        outputs = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            loss = loss_fn(outputs, labels)
        loss.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, inputs + eps), inputs - eps)
        x_adv = torch.max(torch.min(x_adv, (1 - dm) / ds), -dm / ds)

    return x_adv


def _pgd_step(args, X, grad, poison_imgs, tau):
    """PGD Step."""
    epsilon = torch.ones(X.size(0)) * (args.eps / 255.)
    epsilon = epsilon.cuda()

    X_ = X.clone()
    X_.grad = grad

    with torch.no_grad():
        X_ -= tau * torch.sign(X_.grad)
        X_ = torch.min(X_, poison_imgs + epsilon.view(X.size(0), 1, 1, 1))
        X_ = torch.max(X_, poison_imgs - epsilon.view(X.size(0), 1, 1, 1))
        X_ = torch.clamp(X_, min=0, max=1)

    return X_