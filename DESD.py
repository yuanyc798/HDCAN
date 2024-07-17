from functools import partial
import math
import torch
import torch.utils.checkpoint as checkpoint
#from einops import rearrange
#from timm.models.layers import DropPath, trunc_normal_
#from timm.models.registry import register_model
from torch import nn
NORM_EPS = 1e-5

# a=torch.tensor([[ 1,  2,  3,  4,  5],
        		# [ 6,  7,  8,  9, 10],
        		# [11, 12, 13, 14, 15]], dtype=float)
# #nn.Softmax(dim=-1)(a)
# print(a.shape)
# attn = a.softmax(dim=0)
# print(attn)
class ESA(nn.Module):
    """
    Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, head_dim=16, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        #print(self.N_ratio)
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        #print('q',q.shape)
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            #print(x_.shape)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            #print('k',k.shape)
            v = self.v(x_)
            #v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            #v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        #attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)

        #x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #x = self.proj(x)
        #x = self.proj_drop(x)
        return attn,v
class DESD(nn.Module):

    def __init__(self, dim, sr_ratio,out_dim=None, head_dim=16, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = self.dim // head_dim
        self.v = nn.Linear(2*dim, self.dim, bias=qkv_bias)
        self.out_dim = out_dim if out_dim is not None else dim
        self.conv=nn.Conv2d(2*self.num_heads,self.num_heads,kernel_size=1,padding=0)
        
        self.esa1=ESA(dim,sr_ratio=sr_ratio)
        self.esa2=ESA(dim,sr_ratio=sr_ratio)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, xx,yy):
        b, c, h,w= xx.shape
        #print(b, c, C)
        x=xx.reshape(b, c, w*h).transpose(1, 2)
        y=yy.reshape(b, c, w*h).transpose(1, 2)
        
        B, N, C = x.shape
        #print(B, N, C)
        x1,v1=self.esa1(x)
        y1,v2=self.esa2(y)
        
        xy=x1+y1
        cat=torch.cat([x1,y1], axis=1)
        cc=self.conv(cat)
        attn=xy+cc
        #print('aat',x1.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        cv=torch.cat([v1,v2], axis=-1)
        v=self.v(cv)#+v1+v2
        v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #x = self.proj(x)
        #x = self.proj_drop(x)
        x=x.transpose(1, 2).reshape(B,C,h,w)
        #x=xx+yy+x
        return x

