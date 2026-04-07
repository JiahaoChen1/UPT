import torch
import torch.nn as nn
from torch.nn import functional as F

from .clip_text import CLIP_Text
from .peft_vit import Peft_ViT, ViT_Tuner
from .peft_rn import Peft_RN, RN_Tuner
from .classifiers import *
from .peft_resnet import *

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit


class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("CLIP-ViT"):
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_ViT(clip_model.visual, cfg)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)
            if cfg.adj_cof:
                self.cofficient = torch.nn.Parameter(torch.ones(12, 12, 10))
        elif cfg.backbone.startswith("CLIP-RN"):
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_RN(clip_model.visual)
            self.tuner = RN_Tuner(cfg, clip_model.visual, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        # if hasattr(clip_model, 'logit_scale'):
        # cfg.scale = float(clip_model.logit_scale.exp())
        # print(f'relocate the temperature to {cfg.scale}')
        self.add_head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features
    
    def forward(self, image, use_tuner=True, return_feature=False, return_attn=False,
                prototype_loss=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        cofficient = self.cofficient if hasattr(self, 'cofficient') else None
        if return_attn:
            return self.image_encoder(image, tuner, head, cofficient)
        if prototype_loss:
            output = self.image_encoder(image, tuner, head, cofficient)
            return output[-1], output[-2]
        return self.image_encoder(image, tuner, head, cofficient)[-1]


class PeftModelFromViT(nn.Module):
    def __init__(self, cfg, vit_model, num_classes, init_stat=None):
        super().__init__()

        if cfg.backbone.startswith("IN21K-ViT"):
            self.image_encoder = Peft_ViT(vit_model, cfg)
            self.tuner = ViT_Tuner(cfg, vit_model, num_classes)
            if cfg.adj_cof:
                self.cofficient = torch.nn.Parameter(torch.ones(12, 1, 10))
                # self.cofficient = torch.nn.Parameter(torch.ones(12, 12, 207, 10))
            
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)


        # self.add_head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        self.add_head = CosineClassifier(768, num_classes, dtype, **cfg)

    def forward(self, image, use_tuner=True, return_feature=False, return_attn=False,
                prototype_loss=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        cofficient = self.cofficient if hasattr(self, 'cofficient') else None
        if return_attn:
            return self.image_encoder(image, tuner, head, cofficient)
        output = self.image_encoder(image, tuner, head, cofficient)
        return output[-1], output[-2]

class PeftModelFromResNet(nn.Module):
    def __init__(self, cfg, vit_model, num_classes, init_stat=None):
        super().__init__()

        self.image_encoder = ResNet(vit_model, cfg)
        self.image_encoder.setup_prompt()

        self.prompt_embeddings_lr = self.image_encoder.prompt_embeddings_lr
        self.prompt_embeddings_tb = self.image_encoder.prompt_embeddings_tb
        
            
        feat_dim = 2048 #if  'rn152' in cfg.backbone else 512
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

        self.add_head = CosineClassifier(768, num_classes, dtype, **cfg)

    def forward(self, image, use_tuner=True, return_feature=False, return_attn=False,
                prototype_loss=False):
        # tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        # cofficient = self.cofficient if hasattr(self, 'cofficient') else None
        output = self.image_encoder(image, head)
        if return_attn:
            return None, output[-2], output[-1]
        
        return output[-1], output[-2]