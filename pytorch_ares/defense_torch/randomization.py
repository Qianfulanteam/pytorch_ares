import torch
import torch.nn.functional as F
from pytorch_ares.defense_torch.utils import Defend


class Randomization(Defend):
    """
    Randomization: defend the model by apply randomization to the input.
    """
    def __init__(self, model, device,data_name, prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
        """
        Initialize Randomization class.
        Args:
            model: torch model, model to protect
            device: torch.device
            prob: float, the prob of randomization
            crop_lst: float list, the list of crop size
        """
        super(Randomization, self).__init__(model, device)
        self.prob = prob
        self.crop_lst = crop_lst
        self.data_name = data_name
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10

    def defend(self, xs:torch.tensor):
        """
        Apply randomization method to the input.
        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
        Returns:
            out: output samples. 4-dim torch tensor, like [batch, channel, width, height]
        """
        xs_ = self.input_transform(xs).to(self.device)
        xs_ = (xs_ - self.mean_torch) / self.std_torch
        return self.model(xs_)

    def input_transform(self, xs:torch.tensor):
        """
        Apply the transform with the given probability.
        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
        Returns:
            out: output samples. 4-dim torch tensor, like [batch, channel, width, height]
        """
        p = torch.rand(1).item()
        if p <= self.prob:
            out = self.random_resize_pad(xs)
            return out
        else:
            return xs

    def random_resize_pad(self, xs:torch.tensor):
        """
        Resize and pad input image randomly.
        Args:
            xs: input samples. 4-dim torch tensor, like [batch, channel, width, height]
        Returns:
            out: output samples. 4-dim torch tensor, like [batch, channel, width, height]
        """
        rand_cur = torch.randint(low=0, high=len(self.crop_lst), size=(1,)).item()
        crop_size = 1 - self.crop_lst[rand_cur]
        pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
        pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2

        if len(xs.shape) == 4:
            bs, c, w, h = xs.shape
        elif len(xs.shape) == 5:
            bs, fs, c, w, h = xs.shape
        w_, h_ = int(crop_size * w), int(crop_size * h)
        # out = resize(xs, size=(w_, h_))
        out = F.interpolate(xs, size=[w_, h_], mode='bicubic', align_corners=False)
        

        pad_left = int(pad_left * (w - w_))
        pad_top = int(pad_top * (h - h_))
        # dim = [pad_left, pad_top, w - pad_left - w_, h - pad_top - h_]
        out = F.pad(out, [pad_left, w - pad_left - w_, pad_top, h - pad_top - h_], value=0)
        
        # out = pad(out, padding=dim, fill=1, padding_mode='constant')
        return out