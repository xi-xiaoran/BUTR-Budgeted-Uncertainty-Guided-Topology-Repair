import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, k, s):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, k, s, k//2)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H, W

class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h=heads
        self.qkv=nn.Linear(dim, dim*3)
        self.proj=nn.Linear(dim, dim)
        self.scale=(dim//heads)**-0.5
    def forward(self, x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,C//self.h).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(dim=-1)
        out=(attn@v).transpose(1,2).reshape(B,N,C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.n1=nn.LayerNorm(dim)
        self.attn=Attention(dim, heads=heads)
        self.n2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class MiTLite(nn.Module):
    def __init__(self, in_ch=3, dims=(32,64,160,256), depths=(1,1,1,1), heads=(1,2,5,8)):
        super().__init__()
        self.pe1=PatchEmbed(in_ch, dims[0], 7, 4)
        self.pe2=PatchEmbed(dims[0], dims[1], 3, 2)
        self.pe3=PatchEmbed(dims[1], dims[2], 3, 2)
        self.pe4=PatchEmbed(dims[2], dims[3], 3, 2)
        self.b1=nn.ModuleList([Block(dims[0], heads[0]) for _ in range(depths[0])])
        self.b2=nn.ModuleList([Block(dims[1], heads[1]) for _ in range(depths[1])])
        self.b3=nn.ModuleList([Block(dims[2], heads[2]) for _ in range(depths[2])])
        self.b4=nn.ModuleList([Block(dims[3], heads[3]) for _ in range(depths[3])])
    def forward(self, x):
        outs=[]
        x,H,W=self.pe1(x)
        for blk in self.b1: x=blk(x)
        outs.append(x.transpose(1,2).reshape(x.size(0),-1,H,W))
        x,H,W=self.pe2(outs[-1])
        for blk in self.b2: x=blk(x)
        outs.append(x.transpose(1,2).reshape(x.size(0),-1,H,W))
        x,H,W=self.pe3(outs[-1])
        for blk in self.b3: x=blk(x)
        outs.append(x.transpose(1,2).reshape(x.size(0),-1,H,W))
        x,H,W=self.pe4(outs[-1])
        for blk in self.b4: x=blk(x)
        outs.append(x.transpose(1,2).reshape(x.size(0),-1,H,W))
        return outs

class SegFormerLite(nn.Module):
    def __init__(self, in_ch=3, head_out_ch=1):
        super().__init__()
        self.bb = MiTLite(in_ch=in_ch)
        dims=(32,64,160,256)
        self.proj=nn.ModuleList([nn.Conv2d(d, 64, 1) for d in dims])
        self.fuse=nn.Conv2d(64*4, 64, 1)
        self.out=nn.Conv2d(64, head_out_ch, 1)
    def forward(self, x):
        c1,c2,c3,c4=self.bb(x)
        p1=self.proj[0](c1)
        p2=F.interpolate(self.proj[1](c2), size=p1.shape[-2:], mode="bilinear", align_corners=False)
        p3=F.interpolate(self.proj[2](c3), size=p1.shape[-2:], mode="bilinear", align_corners=False)
        p4=F.interpolate(self.proj[3](c4), size=p1.shape[-2:], mode="bilinear", align_corners=False)
        f=self.fuse(torch.cat([p1,p2,p3,p4],1))
        y=self.out(f)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
