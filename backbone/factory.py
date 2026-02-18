from __future__ import annotations
import torch.nn as nn
from .unet import UNet
from .attention_unet import AttentionUNet
from .unetpp import UNetPP
from .segformer import SegFormerLite
from .swin_unet import SwinUNetLite

# Accept common aliases so experiment scripts are forgiving.
_ALIASES = {
    # UNet
    "unet": "unet",
    # Attention-UNet
    "attentionunet": "attention_unet",
    "attention_unet": "attention_unet",
    "attention-unet": "attention_unet",
    "attnunet": "attention_unet",
    "attunet": "attention_unet",
    # UNet++
    "unetpp": "unetpp",
    "unet_pp": "unetpp",
    "unet++": "unetpp",
    # SegFormer (lite)
    "segformer": "segformer",
    "segformer_lite": "segformer",
    "segformer-lite": "segformer",
    # Swin-UNet (lite)
    "swinunet": "swin_unet",
    "swin_unet": "swin_unet",
    "swin-unet": "swin_unet",
    "swin_unet_lite": "swin_unet",
    "swinunet_lite": "swin_unet",
}

def build_backbone(name: str, head_mode: str = "standard", in_ch: int = 3, base: int = 32) -> nn.Module:
    name_in = (name or "").lower()
    name = _ALIASES.get(name_in, name_in)

    head_out_ch = 1 if head_mode == "standard" else 2

    if name == "unet":
        return UNet(in_ch=in_ch, base=base, head_out_ch=head_out_ch)
    if name == "attention_unet":
        return AttentionUNet(in_ch=in_ch, base=base, head_out_ch=head_out_ch)
    if name == "unetpp":
        return UNetPP(in_ch=in_ch, base=base, head_out_ch=head_out_ch)
    if name == "segformer":
        return SegFormerLite(in_ch=in_ch, head_out_ch=head_out_ch)
    if name == "swin_unet":
        return SwinUNetLite(in_ch=in_ch, base=base, head_out_ch=head_out_ch)

    raise ValueError(
        f"Unknown backbone: {name_in}. "
        f"Supported: unet, attention_unet(attunet/attnunet), unetpp(unet++), segformer, swin_unet(swinunet)."
    )
