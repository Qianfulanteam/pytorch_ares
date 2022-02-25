import numpy as np
import timm
import os
os.environ['TORCH_HOME']=os.path.join(os.path.abspath(os.path.dirname(__file__)),'model')
import torch
from pytorch_ares.example.cifar10.pytorch_cifar10.models import *
from pytorch_ares.dataset_torch.datasets_test import datasets
from pytorch_ares.defense_torch import *
from pytorch_ares.attack_torch import *
DEFENSE = {
    'jpeg': Jpeg_compresssion,
    'bit': BitDepthReduction,   
    'random': Randomization,
}

ATTACKS = {
    'fgsm': FGSM,
    'bim': BIM,
    'pgd': PGD,
    'mim': MIM,
    'cw': CW,
    'deepfool': DeepFool,
    'dim': DI2FGSM,
    'tim': TIFGSM,
}

def test(args):
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    if args.dataset_name == "imagenet":
        net = timm.create_model('resnet50', pretrained=True)
        from timm.models.resnet import default_cfgs
        input_size = default_cfgs['resnet50']['input_size'][1]
        crop_pct = default_cfgs['resnet50']['crop_pct']
        interpolation = default_cfgs['resnet50']['interpolation']
        test_loader = datasets(args.dataset_name, args.batchsize,input_size,crop_pct,interpolation, args.cifar10_path, args.imagenet_val_path, args.imagenet_targrt_path,args.imagenet_path)
        net.to(device)
    else:
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model/checkpoint/resnet18_ckpt.pth')
        net = ResNet18()
        pretrain_dict = torch.load(path, map_location=device)
        net.load_state_dict(pretrain_dict['net'])
        net.to(device)
        test_loader = datasets(args.dataset_name, args.batchsize, args.input_size, args.crop_pct, args.interpolation, args.cifar10_path, args.imagenet_val_path, args.imagenet_targrt_path,args.imagenet_path)
    net.eval()
    distortion = 0
    dist= 0
    success_num = 0
    test_num= 0
    
    if args.attack_name == 'fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'bim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'pgd':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss,device=device)
    elif args.attack_name == 'mim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, decay_factor=args.decay_factor, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)

    if args.defense_name == 'jpeg':
        defense_class = DEFENSE[args.defense_name]
        defense = defense_class(net, device,data_name=args.dataset_name, quality=95)
    if args.defense_name == 'bit':
        defense_class = DEFENSE[args.defense_name]
        defense = defense_class(net, device,data_name=args.dataset_name, compressed_bit=4)
    if args.defense_name == 'random':
        defense_class = DEFENSE[args.defense_name]
        defense = defense_class(net, device, data_name=args.dataset_name, prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02])

    
    if args.dataset_name == "imagenet":
        for i, (image,labels,target_labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            adv_image= attack.forward(image, labels, target_labels)
            
            
            out_defense= defense.defend(adv_image)
            
            out = defense.defend(image)
            
            out_defense = torch.argmax(out_defense, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            if args.target:
                success_num +=(out_defense == target_labels).sum()
            else:
                success_num +=(out_defense != labels).sum()
            
            if i % 2 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                print("%s数据集第%d次分类准确率：%.4f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.4f %%" %(args.defense_name, args.dataset_name, i, adv_acc*100))
           
       
        total_num = len(test_loader.dataset)
      
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        
        print("%s数据集分类准确率：%.4f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集防御成功率：%.4f %%" %(args.defense_name, args.dataset_name, success_num*100))

    if args.dataset_name == "cifar10":
        for i, (image,labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            adv_image= attack.forward(image, labels,None)
            
            out_defense= defense.defend(adv_image)
            
            out = defense.defend(image)
            
            out_defense = torch.argmax(out_defense, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            success_num +=(out_defense != labels).sum()

            if i % 2 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                
                print("%s数据集第%d次分类准确率：%.2f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次防御成功率：%.2f %%" %(args.defense_name, args.dataset_name, i, adv_acc*100))

            
        total_num = len(test_loader.dataset)
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集防御成功率：%.2f %%" %(args.defense_name, args.dataset_name, success_num*100))
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data preprocess args 
    parser.add_argument("--gpu", type=str, default="7", help="Comma separated list of GPU ids")
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='')
    parser.add_argument('--dataset_name', default='imagenet', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--norm', default= np.inf, help='You can choose linf and l2', choices=[np.inf, 1, 2])
    parser.add_argument('--batchsize', default=5, help= 'batchsize for this model')
    parser.add_argument('--attack_name', default='fgsm', help= 'Dataset for this model', choices= ['fgsm', 'bim', 'pgd','mim', 'dim', 'tim', 'deepfool', 'cw'])
    parser.add_argument('--defense_name', default='jpeg', help= 'Dataset for this model', choices= ['jpeg','bit', 'random'])
    parser.add_argument('--cifar10_path', default=os.path.join(os.path.abspath(os.path.dirname(__file__)),'data/CIFAR10'), help='cifar10_path for this model')
    parser.add_argument('--imagenet_val_path', default=os.path.join(os.path.abspath(os.path.dirname(__file__)),'data/val.txt'), help='imagenet_val_path for this model')
    parser.add_argument('--imagenet_targrt_path', default=os.path.join(os.path.abspath(os.path.dirname(__file__)),'data/target.txt'), help='imagenet_targrt_path for this model')
    parser.add_argument('--imagenet_path', default=os.path.join(os.path.abspath(os.path.dirname(__file__)),'data/ILSVRC2012_img_val'), help='imagenet_path for this model')
    parser.add_argument('--eps', type= float, default=8/255.0, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/2550.0 , help='linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
    parser.add_argument('--resize_rate', type= float, default=0.9, help='dim is used')
    parser.add_argument('--diversity_prob', type= float, default=0.5, help='dim is used')
    parser.add_argument('--kernel_name', default='gaussian', help= 'kernel_name for tim', choices= ['gaussian', 'linear', 'uniform'])
    parser.add_argument('--len_kernel', type= int, default=15, help='len_kernel for tim')
    parser.add_argument('--nsig', type= int, default=3, help='nsig for tim')
    # parser.add_argument('--n_restarts', type= int, default=1, help='n_restarts for apgd')
    parser.add_argument('--seed', type= int, default=0, help='seed for apgd')
    parser.add_argument('--loss', default='ce', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'dlr'])
    parser.add_argument('--binary_search_steps', type= int, default=10, help='search_steps for cw')
    parser.add_argument('--max_steps', type= int, default=200, help='max_steps for cw')
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])

    args = parser.parse_args()

    test(args)
