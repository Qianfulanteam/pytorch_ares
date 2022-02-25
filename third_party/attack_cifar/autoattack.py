import time
from third_party.attack_cifar.utils import Attack
from third_party.attack_cifar.apgd import APGD, APGDT
from third_party.attack_cifar.fab import FAB
from third_party.attack_cifar.square import Square
from third_party.attack_cifar.multiattack import MultiAttack


class autoAttack(Attack):
    r"""
    AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]
    Distance Measure : Linf, L2
    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model,device,steps,rho,alpha_max,eta,resc_schedule,n_queries,beta,targeted1,p_init,loss1, norm='Linf', eps=.3, version='standard', seed=None, verbose=False):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.beta = beta
        self.n_queries = n_queries
        self.resc_schedule = resc_schedule
        self.p_init = p_init
        self.loss1 = loss1
        self.targeted1 = targeted1
        self.alpha_max = alpha_max
        self.rho = rho
        self.eta =eta
        self.steps = steps
        self.device = device
        self.version = version
        self.seed = seed
        self.verbose = verbose

        if version == 'standard':
            self.autoattack = MultiAttack([
                APGD(model, device=device, norm=norm, eps=eps, steps=steps,n_restarts=1, seed=self.get_seed(), loss='ce', eot_iter=1,rho=rho, verbose=verbose),
                APGDT(model, device=device, norm=norm,  eps=eps, steps=steps, n_restarts=1, seed=self.get_seed(), eot_iter=1,rho=rho, verbose=verbose),
                FAB(model, norm=norm, eps=None, steps=steps,n_restarts=1, alpha_max=alpha_max, eta=eta,beta=beta, verbose=verbose, seed=self.get_seed(), targeted=targeted1, device=device),
                Square(model, device=device, _targeted=targeted1, norm=norm, eps=eps, n_queries=n_queries, n_restarts=1, p_init=p_init, loss=loss1, resc_schedule=resc_schedule, seed=self.get_seed(), verbose=verbose),
            ])

        elif version == 'plus':
            self.autoattack = MultiAttack([
                APGD(model, device=device, norm=norm, eps=eps, steps=steps,n_restarts=5, seed=self.get_seed(), loss='ce', eot_iter=1,rho=rho, verbose=verbose),
                APGD(model, device=device, norm=norm, eps=eps, steps=steps,n_restarts=5, seed=self.get_seed(), loss='dlr', eot_iter=1,rho=rho, verbose=verbose),
                FAB(model, norm=norm, eps=None, steps=steps,n_restarts=5, alpha_max=alpha_max, eta=eta,beta=beta, verbose=verbose, seed=self.get_seed(), targeted=targeted1, device=device),
                Square(model, device=device, _targeted=targeted1, norm=norm, eps=eps, n_queries=n_queries, n_restarts=1, p_init=p_init, loss=loss1, resc_schedule=resc_schedule, seed=self.get_seed(), verbose=verbose),
                APGDT(model, device=device, norm=norm,  eps=eps, steps=steps, n_restarts=1, seed=self.get_seed(), eot_iter=1,rho=rho, verbose=verbose),
                FAB(model, norm=norm, eps=None, steps=steps,n_restarts=1, alpha_max=alpha_max, eta=eta,beta=beta, verbose=verbose, seed=self.get_seed(), targeted=True, device=device),
            ])

        elif version == 'rand':
            self.autoattack = MultiAttack([
                APGD(model, device=device, norm=norm, eps=eps, steps=steps,n_restarts=1, seed=self.get_seed(), loss='ce', eot_iter=20,rho=rho, verbose=verbose),
                APGD(model, device=device, norm=norm, eps=eps, steps=steps,n_restarts=1, seed=self.get_seed(), loss='dlr', eot_iter=20,rho=rho, verbose=verbose),
            ])

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed