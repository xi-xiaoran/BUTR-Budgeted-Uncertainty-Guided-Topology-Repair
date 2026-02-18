import torch
import torch.nn.functional as F

def _soft_erode(img):
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1,p2)

def _soft_dilate(img):
    return F.max_pool2d(img, 3, 1, 1)

def _soft_open(img):
    return _soft_dilate(_soft_erode(img))

def _soft_skel(img, iters):
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel*delta)
    return skel

def soft_cldice_loss(logits, target, iters=20, eps=1e-6):
    p = torch.sigmoid(logits)
    sk_p = _soft_skel(p, iters)
    sk_g = _soft_skel(target, iters)
    tprec = (sk_p*target).sum(dim=(2,3)) / (sk_p.sum(dim=(2,3)) + eps)
    tsens = (sk_g*p).sum(dim=(2,3)) / (sk_g.sum(dim=(2,3)) + eps)
    cl = 2*tprec*tsens/(tprec+tsens+eps)
    return 1.0 - cl.mean()
