import torch
import torch.nn as nn
import torch.nn.functional as F


class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()
    

class LinearClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
        # return F.linear(x, self.weight)


class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale
        # self.cofficient = nn.Parameter(torch.zeros(num_classes, feat_dim, dtype=dtype))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale


class L2NormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
    
    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class ExpertsClassifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=25, **kwargs):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight1.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.weight2 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight2.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.weight3 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight3.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.scale = scale

    def forward(self, x):
        # x1 = F.dropout(x, p=0.1, training=self.training)
        # x2 = F.dropout(x, p=0.1, training=self.training)
        # x3 = F.dropout(x, p=0.1, training=self.training)

        # x1 = F.normalize(x1, dim=-1)
        # x2 = F.normalize(x2, dim=-1)
        # x3 = F.normalize(x3, dim=-1)
        x = F.normalize(x, dim=-1)
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=0)
        weight1 = F.normalize(self.weight1, dim=-1)
        weight2 = F.normalize(self.weight2, dim=-1)
        weight3 = F.normalize(self.weight3, dim=-1)
        if self.training:
            extra_info = {}
            extra_info['head_logits'] = F.linear(x1, weight1) * self.scale
            extra_info['medium_logits'] = F.linear(x2, weight2) * self.scale
            extra_info['few_logits'] = F.linear(x3, weight3) * self.scale
            return extra_info
        else:
            return (F.linear(x1, weight1) + F.linear(x2, weight2) + F.linear(x3, weight3)) * self.scale
