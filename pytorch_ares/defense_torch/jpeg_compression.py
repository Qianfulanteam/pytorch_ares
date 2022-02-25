import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO#BytesIO实现了在内存中读写bytes
from pytorch_ares.defense_torch.utils import Defend


_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()



class Jpeg_compresssion(Defend):

    def __init__(self, model, device,data_name, quality=75):
        super(Jpeg_compresssion, self).__init__(model, device)
        self.quality = quality
        self.data_name = data_name
        if self.data_name=='imagenet':
            self.mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)#imagenet
            self.std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)#imagenet
        else:
            self.mean_torch = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)#cifar10
            self.std_torch = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)#cifar10
    
    def defend(self, x):
        
        xs = self.jpegcompression(x).to(self.device)
        xs = (xs-self.mean_torch) / self.std_torch
        
        return self.model(xs)
    
    
    def jpegcompression(self, x):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=self.quality)#压缩成jpeg
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))
