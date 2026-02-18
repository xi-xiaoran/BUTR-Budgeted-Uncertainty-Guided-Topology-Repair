import torch
import torch.nn.functional as F

def _softplus_evidence(x):
    return F.softplus(x)

def edl_uncertainty_from_evidence(evidence):
    ev = _softplus_evidence(evidence)
    alpha = ev + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    K = alpha.shape[1]
    vacuity = K / (S + 1e-6)
    prob = alpha[:,1:2] / (S + 1e-6)
    return prob, vacuity

def edl_binary_loss(evidence, target, coeff_kl=1e-4):
    ev = _softplus_evidence(evidence)
    alpha = ev + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    prob = alpha / (S + 1e-6)
    T = torch.cat([1.0-target, target], dim=1)
    nll = -(T * torch.log(prob + 1e-6)).sum(dim=1, keepdim=True).mean()
    K = alpha.shape[1]
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    lgamma_sum = torch.lgamma(sum_alpha)
    lgamma = torch.lgamma(alpha).sum(dim=1, keepdim=True)
    digamma = torch.digamma(alpha)
    digamma_sum = torch.digamma(sum_alpha)
    kl = (lgamma_sum - lgamma - torch.lgamma(torch.tensor(float(K), device=alpha.device))
          + (alpha - 1) * (digamma - digamma_sum)).sum(dim=1, keepdim=True).mean()
    return nll + coeff_kl*kl
