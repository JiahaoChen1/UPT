import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=10):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, logit, target):
        index = torch.zeros_like(logit, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        return F.cross_entropy(output, target)


class ClassBalancedLoss(nn.Module):
    def __init__(self, cls_num_list, beta=0.9999):
        super().__init__()
        per_cls_weights = (1.0 - beta) / (1.0 - (beta ** cls_num_list))
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class GeneralizedReweightLoss(nn.Module):
    def __init__(self, cls_num_list, exp_scale=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        per_cls_weights = 1.0 / (cls_num_ratio ** exp_scale)
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

    def forward(self, logit, target):
        logit_adjusted = logit + self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)

class InverseAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        prior = cls_num_list / torch.sum(cls_num_list)
        self.prior = prior.float()
        # self.log_cls_num = log_cls_num
        self.tau = tau
    
    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, logit, target):
        inverse_prior = self.inverse_prior(self.prior)
        logit = logit + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        return F.cross_entropy(logit, target)


class CrossEntropy(nn.Module):
    def __init__(self, cls_num_list) -> None:
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (0.5 / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
    
    def forward(self, logit, target):
        # index = torch.zeros_like(logit, dtype=torch.uint8)
        # index.scatter_(1, target.data.view(-1, 1), 1)
        # logit = torch.where(index, logit-1.0, logit) 
        # index = torch.zeros_like(logit, dtype=torch.uint8)
        # index.scatter_(1, target.data.view(-1, 1), 1)

        # index_float = index.type(torch.cuda.FloatTensor)
        # batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # batch_m = batch_m.view((-1, 1))
        # logit_m = logit - batch_m * 25  # scale only the margin, as the logit is already scaled.
        # logit_m = torch.where(index, logit_m, logit)
        return F.cross_entropy(logit, target)

class LADELoss(nn.Module):
    def __init__(self, cls_num_list, remine_lambda=0.1, estim_loss_weight=0.1):
        super().__init__()
        self.num_classes = len(cls_num_list)
        self.prior = cls_num_list / torch.sum(cls_num_list)

        self.balanced_prior = torch.tensor(1. / self.num_classes).float().to(self.prior.device)
        self.remine_lambda = remine_lambda

        self.cls_weight = cls_num_list / torch.sum(cls_num_list)
        self.estim_loss_weight = estim_loss_weight

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
 
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, logit, target):
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss =  F.cross_entropy(logit_adjusted, target)

        per_cls_pred_spread = logit.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        estim_loss = -torch.sum(estim_loss * self.cls_weight)

        return ce_loss + self.estim_loss_weight * estim_loss

class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=25, tau=2, device=None):
        super().__init__()
        self.base_loss = F.cross_entropy
     
        prior = cls_num_list / torch.sum(cls_num_list)
        self.prior = prior.float()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 


    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, extra_info, target):
        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['head_logits']
        expert2_logits = extra_info['medium_logits']
        expert3_logits = extra_info['few_logits']  

        loss = loss + self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss = loss + self.base_loss(expert2_logits, target)
        
        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss = loss + self.base_loss(expert3_logits, target)

        return loss / 3
    

class DiverseKLLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=25, tau=2, device=None):
        super().__init__()
        self.base_loss = F.cross_entropy
     
        prior = cls_num_list / torch.sum(cls_num_list)
        self.prior = prior.float()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 


    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, extra_info, target):
        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['head_logits']
        expert2_logits = extra_info['medium_logits']
        expert3_logits = extra_info['few_logits']  

        expert1_logits = expert1_logits + torch.log(self.prior + 1e-9) 
        loss = loss + self.base_loss(expert1_logits, target)

        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss = loss + self.base_loss(expert2_logits, target)

        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) 
        loss = loss + self.base_loss(expert3_logits, target)

        # prob_expert1_logits = torch.softmax(expert1_logits, dim=1)
        # prob_expert2_logits = torch.softmax(expert2_logits, dim=1)
        # prob_expert3_logits = torch.softmax(expert3_logits, dim=1)

        # # kl_div1 = F.kl_div(prob_expert1_logits.log(), prob_expert2_logits) + F.kl_div(prob_expert2_logits.log(), prob_expert1_logits)
        # # kl_div2 = F.kl_div(prob_expert2_logits.log(), prob_expert3_logits) + F.kl_div(prob_expert3_logits.log(), prob_expert2_logits)
        # # kl_div3 = F.kl_div(prob_expert1_logits.log(), prob_expert3_logits) + F.kl_div(prob_expert3_logits.log(), prob_expert1_logits)
        # kl_div1 = (prob_expert1_logits - prob_expert2_logits).abs().mean()
        # kl_div2 = (prob_expert1_logits - prob_expert3_logits).abs().mean()
        # kl_div3 = (prob_expert3_logits - prob_expert2_logits).abs().mean()

        return loss / 3 #+ 1 / (kl_div1 + kl_div2 + kl_div3)  * 1e-2
        
    

class GCLLoss(nn.Module):
    
    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul = 1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1/3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
           
                                         
    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
             
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device) #self.scale(torch.randn(cosine.shape).to(cosine.device))  
        
        #cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list   
        cosine = cosine - self.noise_mul * noise.abs()/self.m_list.max() *self.m_list         
        output = torch.where(index, cosine-self.m, cosine)                    
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:    
            return F.cross_entropy(self.s*output, target, weight=self.weight)     

def simi_loss(fd_mus, fd_sigmas, ft_mus, ft_sigmas, feature, base_feature):
    shift_fd_x = (base_feature - fd_mus)
    # print(shift_fd_x)
    fd_score = ((shift_fd_x / fd_sigmas) * shift_fd_x).sum(dim=1)
    shift_ft_x = (feature - ft_mus)
    ft_score = ((shift_ft_x / ft_sigmas) * shift_ft_x).sum(dim=1)

    return (ft_score - fd_score).abs().mean(dim=0)

def log_prob_loss(ft_mus, ft_sigmas, feature):
    coff = torch.log(ft_sigmas + 1e-9).sum(dim=1) * -0.5
    diff = feature - ft_mus
    score = (diff * diff / ft_sigmas).sum(dim=1) * -0.5
    return (coff + score - 0.9198).mean(dim=0)


def update_stat(ft_mus, ft_sigmas, feature, ema=0.1, num_class=-1, label=None):

    batch_size, dim_ = feature.shape
    expand_feature = feature.view(batch_size, 1, dim_).expand(batch_size, num_class, dim_)

    expand_index = torch.zeros((batch_size, num_class)).to(ft_mus.device)
    expand_index.scatter_(1, label.view(-1, 1), 1)
    expand_index = expand_index.view(batch_size, num_class, 1).expand(batch_size, num_class, dim_)

    select_feature = expand_feature * expand_index
    amount_feature = expand_index.sum(dim=0)
    amount_feature[amount_feature == 0] = 1

    avg_feature = select_feature.sum(dim=0) / amount_feature

    ft_mus = ft_mus * (1 - ema) + feature * ema


# class EstimatorCV():
#     def __init__(self, feature_num, class_num):
#         super(EstimatorCV, self).__init__()
#         self.class_num = class_num
#         self.CoVariance = torch.zeros(class_num, feature_num).cuda()
#         self.Ave = torch.zeros(class_num, feature_num).cuda()
#         self.Amount = torch.zeros(class_num).cuda()

def update_CV(features, labels, model, num_classes=-1):
    N = features.size(0)
    C = num_classes
    A = features.size(1)

    NxCxFeatures = features.view(
        N, 1, A
    ).expand(
        N, C, A
    )
    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)

    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1

    ave_CxA = features_by_sort.sum(0) / Amount_CxA

    var_temp = features_by_sort - \
                ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

    var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

    sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

    weight_CV = sum_weight_CV.div(
        sum_weight_CV + model.amount.view(C, 1).expand(C, A)
    )

    weight_CV[weight_CV != weight_CV] = 0

    additional_CV = weight_CV.mul(1 - weight_CV).mul((model.mus - ave_CxA).pow(2))

    model.sigmas = (model.sigmas.mul(1 - weight_CV) + var_temp
                        .mul(weight_CV)).detach() + additional_CV.detach()

    model.mus = (model.mus.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

    model.amount = model.amount + onehot.sum(0)


def cluster(features, base_features, cls_label, num_class, score):

    onehot = cls_label - cls_label[:, None]      # (N, N)
    onehot[onehot != 0] = -1
    onehot[onehot == 0] = 1
    onehot[onehot == -1] = 0

    # onehot_diff = (1 - onehot)

    # onehot_same = onehot * (1 - torch.eye(onehot.shape[0]).to(onehot.device))

    features = F.normalize(features, dim=-1)
    base_features = F.normalize(base_features, dim=-1)

    features = F.linear(features, features)
    base_features = F.linear(base_features, base_features)

    features = features * onehot + -1e9 * (1 - onehot)
    base_features = base_features * onehot + -1e9 * (1 - onehot)
    
    features = torch.log_softmax(features * score, dim=-1)
    base_features = torch.softmax(base_features, dim=-1)

    res = F.kl_div(features, base_features, reduction='none').sum(dim=-1)
 
    return res


def uniformity(features, base_features, cls_label, num_class, score):

    # features = F.dropout(features, p=1.0, training=True)
    # base_features = F.dropout(base_features, p=0.8, training=True)

    N = features.size(0)
    C = num_class
    A1 = features.size(1)
    A2 = base_features.size(1)

    onehot = torch.zeros(N, C).to(features.device)
    onehot.scatter_(1, cls_label.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1)#.expand(N, C, A)
    Amount_CxA = NxCxA_onehot.sum(0)

    valid = (Amount_CxA.sum(dim=-1) != 0)

    base_features = base_features.view(
        N, 1, A2
    ).expand(
        N, C, A2
    )
    base_features = base_features.mul(NxCxA_onehot)
    Amount_CxA[Amount_CxA == 0] = 1
    ave_base = base_features.sum(0) / Amount_CxA


    features = features.view(
        N, 1, A1
    ).expand(
        N, C, A1
    )
    features = features.mul(NxCxA_onehot)
    Amount_CxA[Amount_CxA == 0] = 1
    ave_fine = features.sum(0) / Amount_CxA

    avg_base = ave_base[valid]
    avg_fine = ave_fine[valid]
    
    # mask = F.dropout(torch.ones(A1), p=0.8, training=True)
    # avg_fine = avg_fine * mask.to(avg_fine.device)
    # avg_fine = avg_fine + torch.randn_like(avg_fine).to(avg_fine.device)
    # score = score[valid]
    # w_valid = weights[valid]

    avg_base = F.normalize(avg_base, dim=-1)
    avg_fine = F.normalize(avg_fine, dim=-1)

    base_score = F.linear(avg_base, avg_base)
    base_score = torch.softmax(base_score, dim=-1)
    # print(base_score)
                                                                                   
    fine_score = F.linear(avg_fine, avg_fine)
    fine_score = torch.log_softmax(fine_score * score , dim=-1)

    divergency = F.kl_div(fine_score, base_score, reduction='none').sum(dim=-1)

    return divergency

class ReverseLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        prior = cls_num_list / torch.sum(cls_num_list)
        self.prior = prior.float()
        self.C_number = len(cls_num_list)  # class number
        self.tau = tau 

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, logit, target):
        inverse_prior = self.inverse_prior(self.prior)
        logit = logit + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 

        return F.cross_entropy(logit, target)