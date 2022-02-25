import torch
import torch.nn.functional as F
import numpy as np

class multitarget(object):
    def __init__(self,model,epsilon, alpha, attack_iters,device, restarts=20,
                num_cla=10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device
        self.attack_iters = attack_iters
        self.restarts = restarts
        self.num_cla = num_cla
        self.lower_limit=0
        self.upper_limit = 1

    ### Multi-target attack
    def multitarget_loss(self,x, y, index=0, num_cla=10):# index =0, 1, ..., num_cla-2
        y_k = index * torch.ones_like(y).to(self.device)
        y_k[y_k >= y] = y_k[y_k >= y] + 1
        loss_value = - x[np.arange(x.shape[0]), y] + x[np.arange(x.shape[0]), y_k]
        return loss_value.mean()


    def clamp(self,X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def forward(self,X, y):
        max_loss = torch.zeros(y.shape[0]).to(self.device)
        max_delta = torch.zeros_like(X).to(self.device)
        for _ in range(self.restarts):
            for ind in range(self.num_cla - 1):
                X_adv = X.clone().detach()
                delta = torch.zeros_like(X).uniform_(-self.epsilon, self.epsilon).to(self.device)
                #upper_limit, lower_limit = 1,0
                delta = self.clamp(delta, self.upper_limit-X, self.lower_limit-X)
                delta = delta.clone().detach().requires_grad_(True)
                for _ in range(self.attack_iters):
                    output = self.model(X_adv + delta)
                    loss = self.multitarget_loss(output, y, index=ind, num_cla=self.num_cla)
                    grad = torch.autograd.grad(loss, delta)[0]
                    d = torch.clamp(delta + self.alpha * torch.sign(grad), min=-self.epsilon, max=self.epsilon)
                    d = self.clamp(d, self.lower_limit - X_adv, self.upper_limit - X_adv)
                    delta.data = d
                output = self.model(X_adv + delta)
                all_loss = F.cross_entropy(output, y, reduction='none')
                max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
                max_loss = torch.max(max_loss, all_loss)
        return X + max_delta   