import torch

class Defend():
    """
    Basic Defend Class
    """
    def __init__(self, model, device):
        """
        Initializing basic defend class object.
        Args:
            model: torch model
            device: torch.device
        """
        self.model = model.to(device)
        self.device = device
        self.loss_lst = {}

    def config_l2(self):
        """
        Initialize l2 loss config.
        Returns:
        """
        self.loss_lst['l2'] = []

    def config_l2_loss(self, x0:torch.tensor, x1:torch.tensor):
        """
        calculate the l2 dis of x0 tensor and x1 tensor.
        Args:
            x0: torch.tensor
            x1: torch.tensor
        Returns:
            None
        """
        l2_loss = torch.square(x0 - x1)
        self.loss_lst['l2'].append(l2_loss)
