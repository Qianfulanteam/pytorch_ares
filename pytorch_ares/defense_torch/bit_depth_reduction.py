import torch
from pytorch_ares.defense_torch.utils import Defend



class BitDepthReduction(Defend):
    """
    BitDepthReduction: Bit-Depth Reduction
    """
    def __init__(self, model, device,data_name, compressed_bit=4):
        """
        Initialize the bit-depth-reduction class.
        Args:
            model: original model
            device: torch.device
            compressed_bit: int, compressed bit
        """
        super(BitDepthReduction, self).__init__(model, device)
        self.compressed_bit = compressed_bit
        self.data_name = data_name
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10

    def defend(self, xs: torch.tensor):
        """
        Defend the attack by bit-depth-reduction method.
        The input will be torch.tensor, and it will be transformed by bit-depth-reduction first, then the input will be feed
        to the original model. This defend method will return the output.
        Args:
            xs: torch.tensor, n-dim
        Returns:
            output: torch.tensor, n-dim
        """
        # compress tensor
        xs_compress = self.bit_depth_reduction(xs)
        xs_compress = (xs_compress - self.mean_torch) / self.std_torch

        # model forward
        output = self.model(xs_compress)

        return output

    def bit_depth_reduction(self, xs: torch.tensor):
        """
        This method implements bit_depth_reduction.
        The main idea of this method is to reduce perturbation of attack by reduce image bit precises.
        Args:
            xs: torch.tensor, n-dim
        Returns:
            xs_compress: torch.tensor, n-dim
        """
        # [0, 1] to [0, 2^compressed_bits-1]
        bits = 2 ** self.compressed_bit #2**i
        xs_compress = (xs.detach() * bits).int()

        # [0, 2^compressed_bit-1] to [0, 255]
        xs_255 = (xs_compress * (255 / bits))

        xs_compress = (xs_255 / 255).to(self.device)

        return xs_compress