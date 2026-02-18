import torch

def _ste(prob, thr=0.5):
    hard = (prob > thr).float()
    return hard + (prob - prob.detach())

def topo_loss_ph_proxy(logits, target, thr=0.5):
    # proxy: mismatch of fragmentation/holes proxies
    prob = torch.sigmoid(logits)
    hard = _ste(prob, thr)
    gt = (target > 0.5).float()
    # fragmentation proxy: total variation / area
    tv = (hard[:,:,:,1:] - hard[:,:,:,:-1]).abs().sum(dim=(2,3)) + (hard[:,:,1:,:] - hard[:,:,:-1,:]).abs().sum(dim=(2,3))
    tvg = (gt[:,:,:,1:] - gt[:,:,:,:-1]).abs().sum(dim=(2,3)) + (gt[:,:,1:,:] - gt[:,:,:-1,:]).abs().sum(dim=(2,3))
    frag = tv / (hard.sum(dim=(2,3)) + 1e-6)
    fragg = tvg / (gt.sum(dim=(2,3)) + 1e-6)
    # hole proxy: count local cavities by Laplacian-of-mask magnitude
    lap = (hard[:,:,1:-1,1:-1]*4 - hard[:,:,1:-1,:-2] - hard[:,:,1:-1,2:] - hard[:,:,:-2,1:-1] - hard[:,:,2:,1:-1]).abs().mean(dim=(2,3))
    lapg = (gt[:,:,1:-1,1:-1]*4 - gt[:,:,1:-1,:-2] - gt[:,:,1:-1,2:] - gt[:,:,:-2,1:-1] - gt[:,:,2:,1:-1]).abs().mean(dim=(2,3))
    return (frag - fragg).abs().mean() + 0.5*(lap - lapg).abs().mean()
