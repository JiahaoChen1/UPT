import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224
import timm

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler, ClassAwareSampler
from utils.losses import *
from utils.evaluator import Evaluator, knn_predict
from PIL import Image, ImageOps, ImageFilter
import random
from operator import mul
from functools import reduce
from utils.gsw import GSW

from thop import profile


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
         
def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    if backbone_name == 'ViT-B/16':
        model_path = '/data00/jiahao/PEL-main/pretrained/ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        # CLIP's default precision is fp16
        model.float()
    # print(model.logit_scale.exp())
    return model


def zero_moduel(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def load_vit_to_cpu(cfg, teacher=False):
    backbone_name = cfg.backbone
    if teacher:
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16":
        # model = vit_base_patch16_224(pretrained=True).eval()
        # load_checkpoint(model, '/data00/jiahao/PEL-main/pretrained/ViT-B-16-imagenet21k.npz')
        # load_checkpoint(model, '/data00/jiahao/l2p-pytorch-main/VLM-l2p/pretrained_model/ViT-B_16.npz')
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True).eval()
        # model = timm.create_model('vit_small_patch16_224_in21k', pretrained=True).eval()

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model

def load_resnet_to_cpu(cfg):
    backbone_name = cfg.backbone
    if backbone_name == "imagenet_sup_rn50":
        model = models.resnet50(pretrained=True)
    elif backbone_name == "imagenet_sup_rn101":
        model = models.resnet101(pretrained=True)  # 2048
    elif backbone_name == "imagenet_sup_rn152":
        model = models.resnet152(pretrained=True)  # 2048
    elif backbone_name == "imagenet_sup_rn34":
        model = models.resnet34(pretrained=True)   # 512
    elif backbone_name == "imagenet_sup_rn18":
        model = models.resnet18(pretrained=True)   # 512
    return model

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class ProjHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(768, 768)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(768, 768))
        ]))
        self.ln_1 = LayerNorm(768)

        # self.mlp2 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(768, 768)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(768, 768))
        # ]))
        self.ln_2 = LayerNorm(768)

        # self.mlp3 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(768, 768)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(768, 768))
        # ]))
        # self.ln_3 = LayerNorm(768)
    
    def forward(self, x):
        # x = F.dropout(x, p=0.5, training=True)
        x = self.mlp1(self.ln_1(x))
        x = self.ln_2(x)
        x = F.relu(x)
        return x


def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
    

class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        if self.cfg.reg:
            self.model_proj = ProjHead().to(self.device)
            # self.model_proj = zero_moduel(self.model_proj)
            # print(self.model_proj)
            pass
        # self.model_proj = ProjHead() 
        self.build_data_loader()
        if cfg.reg:
            self.preprocess()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        
        self._writer = None
        self.gsw = GSW()
        
        
        # cfg = self.cfg
        # classnames = self.classnames

        # print("Initialize head with text features")
        # with torch.no_grad():
            # prompts = self.get_tokenized_prompts(self.classnames)
            # text_features = self.model.encode_text(prompts)
            # text_features = F.normalize(text_features, dim=-1)

            # if cfg.backbone.startswith("CLIP-ViT"):
            #     text_features = text_features @ self.model.image_encoder.proj.t()
            #     text_features = F.normalize(text_features, dim=-1)

            # self.base_classifier = text_features
        # zeroshot_clip_model = load_clip_to_cpu(cfg)
        # self.zeroshot_clip_model = ZeroShotCLIP(zeroshot_clip_model)
        # self.zeroshot_clip_model.to(self.device)
        # prompts = self.get_tokenized_prompts(self.classnames)
        # self.zeroshot_clip_model.init_text_features(prompts)

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP") or cfg.backbone.startswith("imagenet"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        if ('CIFAR' in cfg.dataset):
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # transforms.RandomErasing(p=0.5, value='random')
            ])
        elif 'ImageNet' in cfg.dataset:
            transform_train = transforms.Compose([
                # transforms.RandAugment(),
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
                # transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # transforms.RandomErasing(p=0.5, value='random')
            ])
            print('use imagenet')
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
                # transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # transforms.RandomErasing(p=0.5, value='random')
            ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.test_ensemble:
            transform_test = transforms.Compose([
                transforms.Resize(resolution + expand),
                transforms.FiveCrop(resolution),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Normalize(mean, std),
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        # test_dataset = datasets.Sketch(root='/data00/jiahao/data/imagenetv2-matched-frequency-format-val', transform=transform_test)
        # test_dataset = datasets.CorruptCIFAR(name=cfg.sub_name, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        if self.cfg.reg:
            # transform_aug = transform_train = transforms.Compose([
            #                 # transforms.RandAugment(),
            #                 # transforms.RandomResizedCrop(resolution),
            #                 transforms.Resize(256),
            #                 transforms.RandomCrop(224),
            #                 transforms.RandomHorizontalFlip(),
            #             #     transforms.RandomApply(
            #             #     [transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #             #                             saturation=0.2, hue=0.1)],
            #             #     p=0.8
            #             # ),
            #                 # transforms.RandomGrayscale(p=0.2),
            #                 # GaussianBlur(p=0.1),
            #                 # Solarization(p=0.2),
            #                 transforms.ToTensor(),
            #                 transforms.Normalize(mean, std),
            #                 # transforms.RandomErasing(p=1.0, value='random'),
            #             ])
            simi_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)

        if self.cfg.knn_only:
            if 'CIFAR' in cfg.dataset:
                knn_dataset = datasets.CIFAR100('/data00/jiahao/data/longtailed_dataset/CIFAR100', train=True, transform=transform_plain)
            else:
                knn_dataset = datasets.Val_Places_LT(root, train=True, transform=transform_train)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=512, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=512, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=128, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        if self.cfg.reg:
            cbs_sampler = ClassAwareSampler(simi_dataset, num_samples_cls=4)
            self.simi_loader = DataLoader(simi_dataset, 
                                          batch_size=cfg.micro_batch_size, num_workers=cfg.num_workers, 
                                          pin_memory=True, sampler=cbs_sampler)
        if self.cfg.knn_only:
            self.knn_loader = DataLoader(knn_dataset,
                batch_size=512, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            prompts = self.get_tokenized_prompts(classnames)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head
            self.add_head = self.model.add_head

            # self.cfg.scale
        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg)
            # if self.cfg.reg:
            #     self.model = PeftModelFromViT(cfg, vit_model, num_classes, init_stat=[self.mus, self.sigmas])
            # else:
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

            self.add_head = self.model.add_head
        
        elif cfg.backbone.startswith("imagenet"):
            print(f"Loading resnet (backbone: {cfg.backbone})")
            resnet_model = load_resnet_to_cpu(cfg)
            self.model = PeftModelFromResNet(cfg, resnet_model, num_classes)
            self.model.to(self.device)
            self.head = self.model.head

            self.tuner1 = self.model.prompt_embeddings_lr
            self.tuner2 = self.model.prompt_embeddings_tb
            self.add_head = self.model.add_head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()

            # self.init_prompt()
        
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")

        if hasattr(self, 'tuner'):
            for name, param in self.tuner.named_parameters():
                param.requires_grad_(True)
        else:
            self.tuner1.requires_grad_(True)
            self.tuner2.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)
        
        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters()) if hasattr(self, 'tunner') else 0
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        # self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
        #                               {"params": self.head.parameters()}],
        #                               lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        if self.cfg.adj_cof:
            self.model.cofficient.requires_grad_(True)
            self.optim = torch.optim.SGD([{"params": self.model.cofficient}],
                                        lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
            
            print("Turning off gradients in the tuner")
            for name, param in self.tuner.named_parameters():
                param.requires_grad_(False)
            print("Turning off gradients in the head")
            for name, param in self.head.named_parameters():
                param.requires_grad_(False)
        else:
            if self.cfg.reg:
                if hasattr(self, 'tuner'):
                    self.optim = torch.optim.SGD([{"params": self.tuner.parameters(),},
                                            {"params": self.head.parameters(),},
                                            {"params": self.model_proj.parameters()},
                                            {"params": self.add_head.parameters()}],
                                            lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
                else:
                    self.optim = torch.optim.SGD([{"params": self.tuner1,},
                                                    {"params": self.tuner2,},
                                            {"params": self.head.parameters(),},
                                            {"params": self.model_proj.parameters()},
                                            {"params": self.add_head.parameters()}],
                                            lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
            else:
                self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                            {"params": self.head.parameters()}],
                                            #   {"params": self.head.cofficient}],
                                            lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        
        # if self.cfg.reg:
        #     self.model.mus.requires_grad_(True)
        #     self.model.sigmas.requires_grad_(True)

        #     self.stat_optim = torch.optim.SGD([{"params": self.model.mus},
        #                                        {"params": self.model.sigmas}],
        #                                 lr=1e-5, weight_decay=cfg.weight_decay, momentum=cfg.momentum)

        
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = CrossEntropy(cls_num_list=cls_num_list)
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=1)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "Reverse": # https://arxiv.org/abs/2012.00321
            self.criterion = ReverseLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GCL": # https://arxiv.org/abs/2012.00321
            self.criterion = GCLLoss(cls_num_list=cls_num_list, m=0.1, s=20, noise_mul=0.5, weight=per_cls_weights).cuda(self.cfg.device)     
        elif cfg.loss_type == "SADE": # https://arxiv.org/abs/2012.00321
            if 'Places' in cfg.dataset:
                self.criterion = DiverseExpertLoss(cls_num_list=cls_num_list, tau=1)
                print('tau equals one')
            else:
                self.criterion = DiverseExpertLoss(cls_num_list=cls_num_list, tau=1)
        elif cfg.loss_type == "DIVKL":
            self.criterion = DiverseKLLoss(cls_num_list=cls_num_list, tau=3)
        
        if cfg.reg:
            beta = 0.9
            effective_num = 1.0 -  cls_num_list ** beta
            weights = (1.0 - beta) / effective_num
            self.eff_weights = (weights / torch.sum(weights) * len(cls_num_list))
            print(f'*******effective weights*******')
            print(self.eff_weights)
            print(cls_num_list)
            # assert 0
            print(f'*******************************')
            self.reverse_criterion = ReverseLoss(cls_num_list=cls_num_list, tau=1)
            # cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
            # log_cls_num = torch.log(cls_num_ratio)
            # self.eff_weights = log_cls_num

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        prompts = self.get_tokenized_prompts(classnames)
        text_features = self.model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_prompt(self):
        print("Initialize the visual prompts")
        inits = []
        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            init_value = self.model(image, use_tuner=False, return_feature=True, init_prompt=True)
            init_value = torch.stack(init_value, dim=0)
            init_value = torch.mean(init_value, dim=2)
            inits.append(init_value)
        inits = torch.cat(inits, dim=1)
        inits = torch.mean(inits, dim=1)

        for idx, v in enumerate(self.tuner.vpt_list):
            # init_val = math.sqrt(6. / float(3 * reduce(mul, v.patch_size, 1) + v.emb_dim))
            v.prompt.data = v.prompt.data + inits[idx]


    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        

        # if self.cfg.reg:
        loss_dis_meter = AverageMeter(ema=True)
        # loss_prob_meter = AverageMeter(ema=True)

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train() if hasattr(self, 'tunner') else None
            end = time.time()

            num_batches = len(self.train_loader)
            cbs_iter = iter(self.simi_loader) if self.cfg.reg else None
            loss_meter = AverageMeter(ema=True)
            acc_meter = AverageMeter(ema=True)
            cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]
            
            # cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)
            # if epoch_idx <= 7:
            #     self.criterion = CrossEntropy(cls_num_list=cls_num_list)
            # else:
            #     self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
            # else:
            #     self.criterion = InverseAdjustedLoss(cls_num_list=cls_num_list)


            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)
                if self.cfg.reg:
                    # cbs_data = next(cbs_iter)
                    # cbs_image, cbs_label = cbs_data[0].to(self.device), cbs_data[1].to(self.device)
                    pass

                if cfg.prec == "amp":
                    with autocast():
                        # loss_dis, base_feature = self.simi_loss(image, label, feature, output)
                        # with torch.no_grad():
                        #     macs, params = profile(self.model, inputs=(image, ))
                        #     print(macs)
                        # assert 0
                        output, feature = self.model(image, prototype_loss=True)
                        # feature = feature + torch.randn_like(feature)  * 0.5
                        
                        # with torch.no_grad():
                        #     _, _, base_feature = self.base_model(image)
                        #     base_feature = self.model_proj(base_feature)

                        # 
                        with torch.no_grad():
                            alpha = 0.2
                            if 'CIFAR' in self.cfg.dataset:
                                alpha = 1.0
                            mixed_x, y_a, y_b, lam = mixup_data(image, label, alpha=alpha)

                            _, _, base_feature = self.base_model(mixed_x)
                            # _, _, o_base_feature = self.base_model(image)
                        o_base_feature = base_feature
                        base_feature = self.model_proj(base_feature) #* base_feature #+ base_feature
                        
                        # label_matrix = torch.zeros((base_feature.shape[0], self.num_classes)).to(base_feature.device).scatter_(1, label.view(-1, 1), 1)
                        # label_matrix[:, self.many_idxs] = 0
                        # selected_sample_idx = (torch.sum(label_matrix, dim=-1) == 1)
 

                        output = self.model.head(feature) #+ torch.randn_like(feature) * 0.5
                        # augment_feat = feature * (epoch_idx / num_epochs)  + (o_base_feature.detach()) * (1 - epoch_idx / num_epochs)
                        # mask = torch.rand_like(o_base_feature)
                        # ratio = epoch_idx / num_epochs
                        # o_base_feature = o_base_feature * (mask > ratio).float() + feature * (mask <= ratio).float()
                        # output2 = self.model.head(o_base_feature.detach())
                        loss = self.criterion(output, label)

                        # base_feature = F.normalize(base_feature, dim=-1)
                        # base_weight = F.normalize(self.base_classifier, dim=-1)
                        # base_output = F.linear(base_feature, base_weight) * 25
                        # loss_base = self.criterion(base_output, label)
                        # loss = loss + loss_base

                        

                        loss_micro = loss / self.accum_step
                        if self.cfg.reg:
                            # loss_dis = self.simi_loss(cbs_image, cbs_label)
                            loss_dis = self.simi_loss(base_feature, label, feature, y_a, y_b, lam, epoch_idx, num_epochs)
                            # if epoch_idx <= 8:
                            #     loss_dis = loss_dis
                            # else:
                            #     loss_dis = loss_dis * 0
                            self.scaler.scale(loss_micro+loss_dis).backward()
                        else:
                            loss_dis = torch.tensor([0.])
                            self.scaler.scale(loss_micro).backward()
                            # print(self.model.cofficient.grad)
                        # loss_prob.backward()
                        # print(self.model.mus.grad)
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                        # if self.cfg.reg:
                        #     self.stat_optim.step()
                        #     self.stat_optim.zero_grad()

                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    if isinstance(output, dict):
                        output = output['head_logits'] + output['medium_logits'] + output['few_logits'] 
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                # if self.cfg.reg:
                loss_dis_meter.update(loss_dis.item())
                    # loss_prob_meter.update(loss_prob.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"loss_dis {loss_dis_meter.val:.4f} ({loss_dis_meter.avg:.4f})"]
                    # info += [f"loss_prob {loss_prob_meter.val:.4f} ({loss_prob_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        if hasattr(self, 'tuner'):
            self.save_model(cfg.output_dir)

        self.test()

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if hasattr(self, 'tuner'):
            if self.tuner is not None:
                self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        attn_list = []
        feature_list = []
        gt_list = []
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            attn_weight, feature, output = self.model(image, return_attn=True)


            output = output #+ base_output
            
            output = output.view(_bsz, _ncrops, -1).mean(dim=1)

            self.evaluator.process(output, label)

            feature = feature.cpu()
            # print(feature.shape)
            feature_list.append(feature)
            gt_list.append(label.cpu())

        results = self.evaluator.evaluate()

        gt_list = torch.cat(gt_list, dim=0)
        feature_list = torch.cat(feature_list, dim=0)

        np.save('cifar_vpt_gt.npy', gt_list.numpy())
        np.save('cifar_vpt_feature.npy', feature_list.numpy())

        # assert 0

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]
    

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }
        if self.cfg.adj_cof:
            checkpoint = {
                "tuner": tuner_dict,
                "head": head_dict,
                "cof": self.model.cofficient
            }
        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]
        if self.cfg.adj_cof and (self.cfg.test_only or self.cfg.test_train):
            self.model.cofficient = checkpoint["cof"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict)
        self.head.load_state_dict(head_dict)
    
    def simi_loss(self, base_feature, cbs_label, feature, y_a, y_b, lam, epoch_idx, num_epochs):
        
        # with torch.no_grad():
        #     alpha = 0.2
        #     if 'CIFAR' in self.cfg.dataset:
        #         alpha = 1.0
        #     mixed_x, y_a, y_b, lam = mixup_data(cbs_image, cbs_label, alpha=alpha)
        #     # cbs_image = transforms.RandomErasing(p=0.5, value='random')(cbs_image)
        #     _, _, base_feature = self.base_model(mixed_x)
            
        # base_feature = self.model_proj(base_feature) #* base_feature #+ base_feature
        base_output = self.add_head(base_feature)

        # cls_num_ratio = torch.tensor(self.cls_num_list) / torch.sum(torch.tensor(self.cls_num_list))
        # log_cls_num = torch.log(cls_num_ratio).to(base_feature.device)
        # base_output = base_output
        # score = score + log_cls_num.unsqueeze(0)
        # loss2 = F.kl_div(torch.log_softmax(base_output, dim=-1), torch.softmax(score.detach(), dim=-1), reduction='batchmean')
        # loss2 = self.criterion(base_output, cbs_label)
        loss2 = mixup_criterion(F.cross_entropy, base_output, y_a, y_b, lam)

        if 1:
            base_feature = F.normalize(base_feature, dim=-1)
            feature = F.normalize(feature, dim=-1)

            if base_feature.shape[0] != feature.shape[0]:

                label_matrix1 = torch.zeros((base_feature.shape[0], self.num_classes)).to(base_feature.device).scatter_(1, cbs_label.view(-1, 1), 1)
                label_matrix1[:, self.many_idxs] = 0
                selected_sample_idx1 = (torch.sum(label_matrix1, dim=-1) == 1)

                label_matrix2 = torch.zeros((base_feature.shape[0], self.num_classes)).to(base_feature.device).scatter_(1, cbs_label.view(-1, 1), 1)
                label_matrix2[:, self.med_idxs] = 0
                selected_sample_idx2 = (torch.sum(label_matrix2, dim=-1) == 1)

                label_matrix3 = torch.zeros((base_feature.shape[0], self.num_classes)).to(base_feature.device).scatter_(1, cbs_label.view(-1, 1), 1)
                label_matrix3[:, self.few_idxs] = 0
                selected_sample_idx3 = (torch.sum(label_matrix3, dim=-1) == 1)

                base_dis1 = F.linear(base_feature[selected_sample_idx1], base_feature).flatten()[:, None] * self.cfg.temper
                base_dis1 = torch.clamp(base_dis1, -1, 1)

                base_dis2 = F.linear(base_feature[selected_sample_idx2], base_feature).flatten()[:, None] * self.cfg.temper
                base_dis2 = torch.clamp(base_dis2, -1, 1)

                base_dis3 = F.linear(base_feature[selected_sample_idx3], base_feature).flatten()[:, None] * self.cfg.temper
                base_dis3 = torch.clamp(base_dis3, -1, 1)

                feature1, feature2, feature3 = torch.chunk(feature, chunks=3, dim=0)
                dis1 = F.linear(feature1[selected_sample_idx1], feature1).flatten()[:, None]
                dis2 = F.linear(feature2[selected_sample_idx2], feature2).flatten()[:, None]
                dis3 = F.linear(feature3[selected_sample_idx3], feature3).flatten()[:, None]

                loss_unif = (self.gsw.gsw(dis1, base_dis1.detach()) + self.gsw.gsw(dis2, base_dis2.detach()) +  self.gsw.gsw(dis3, base_dis3.detach())) / 3
            else:
                # print('ok')
                label_matrix = torch.zeros((base_feature.shape[0], self.num_classes)).to(base_feature.device).scatter_(1, cbs_label.view(-1, 1), 1)
                label_matrix[:, self.many_idxs] = 0
                selected_sample_idx = (torch.sum(label_matrix, dim=-1) == 1)

                base_dis = F.linear(base_feature[selected_sample_idx], base_feature).flatten()[:, None] * self.cfg.temper
                base_dis = torch.clamp(base_dis, -1, 1)

                dis = F.linear(feature[selected_sample_idx], feature).flatten()[:, None]   
                loss_unif = ((dis - base_dis.detach()) ** 2).mean()
                # loss_unif = self.gsw.gsw(dis, base_dis.detach())
        return self.cfg.weight * loss_unif + loss2


    @torch.no_grad()
    def preprocess(self):
        if self.cfg.backbone.startswith("CLIP"):
            base_model = load_clip_to_cpu(self.cfg).visual
            # self.model = Peft_ViT(cfg, clip_model, num_classes)
        else:
            base_model = load_vit_to_cpu(self.cfg, teacher=True)
        self.base_model = Peft_ViT(base_model, self.cfg).to(self.device)
        self.base_model.eval()
        print('init base model successfully...')

    @torch.no_grad()
    def knn_test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        feature_bank, target_bank = [], []

        for batch in tqdm(self.knn_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            # _bsz, _ncrops, _c, _h, _w = image.size()
            # image = image.view(_bsz * _ncrops, _c, _h, _w)

            # _, _, feature = self.base_model(image)
            attn_weight, feature, output = self.model(image, return_attn=True)
        
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(label)
    
        feature_bank = torch.cat(feature_bank, dim=0).t()
        target_bank = torch.cat(target_bank, dim=0)

        
        data_loader = self.test_loader
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            # _, _, feature = self.base_model(image)
            attn_weight, feature, output = self.model(image, return_attn=True)
            feature = F.normalize(feature, dim=1)
            pred = knn_predict(feature, feature_bank, target_bank, classes=self.num_classes, knn_k=200, knn_t=0.1)
            self.evaluator.process(pred, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]