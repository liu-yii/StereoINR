
from math import sqrt
from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models


from collections import namedtuple
import cupy
from string import Template

from einops import rearrange
from einops.layers.torch import Rearrange
import numbers
import math
import numpy as np
from einops import rearrange, repeat

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def make_coord(shape):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # v0, v1 = -1, 1

        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    ret = torch.stack(torch.meshgrid(coord_seqs), dim=-1)
    return ret


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor):
        return torch.sin(self.w0 * x)
    

def sine_init(m, w0=30.0, first_layer=False):
    if isinstance(m, nn.Linear):
        in_dim = m.weight.size(1)
        if first_layer:
            m.weight.data.uniform_(-1 / in_dim, 1 / in_dim)
        else:
            bound = math.sqrt(6 / in_dim) / w0
            m.weight.data.uniform_(-bound, bound)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act='relu', w0=30.0):
        super().__init__()

        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU(True) 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'siren':
            self.act = Sine()
        else:
            assert False, f'activation {act} is not supported'

        layers = []
        lastv = in_dim

        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden

        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        if act.lower() == 'siren':
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    sine_init(layer, w0, first_layer=(i == 0))

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
    



class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class cross_simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.q_proj = nn.Conv2d(midc, midc, 1)
        self.k_proj = nn.Conv2d(midc, midc, 1)
        self.v_proj = nn.Conv2d(midc, midc, 1)

        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
    
    def forward(self, query, key, value, name='0'):
        B, C, H, W = query.shape
        bias = query
        
        q = self.q_proj(query).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, self.headc).permute(0, 2, 1, 3)
        k = self.k_proj(key).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, self.headc).permute(0, 2, 1, 3)
        v = self.v_proj(value).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, self.headc).permute(0, 2, 1, 3)

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class crossfusion(nn.Module):
    def __init__(self, midc):
        super().__init__()
        # # spatial attention
        self.spatial_attn = nn.Conv2d(
            midc * 2, midc * 2, 1)
        self.spatial_attn_mul1 = nn.Conv2d(
            midc * 2, midc * 2, 3, padding=1)
        self.spatial_attn_mul2 = nn.Conv2d(
            midc * 2, midc, 3, padding=1)
        self.spatial_attn_add1 = nn.Conv2d(
            midc * 2, midc * 2, 3, padding=1)
        self.spatial_attn_add2 = nn.Conv2d(
            midc * 2, midc, 3, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Sequential(
                nn.Conv2d(midc * 2, midc, 1),
                nn.GELU(),
                nn.Conv2d(midc, midc, 1)
            )
    
    def forward(self, raw, align):
        B, C, H, W = raw.shape

        attn = self.lrelu(self.spatial_attn(torch.cat([raw, align], dim=1)))
        attn_mul = self.spatial_attn_mul2(self.lrelu(self.spatial_attn_mul1(attn)))
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn_mul = torch.sigmoid(attn_mul)

        ret = align * attn_mul + attn_add
        res = self.fc(torch.cat([raw, ret], dim=1)) + raw
        
        return res

class cross_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.q_proj = nn.Linear(midc, midc, 1)
        self.k_proj = nn.Linear(midc, midc, 1)
        self.v_proj = nn.Linear(midc, midc, 1)

        self.o_proj1 = nn.Linear(midc, midc, 1)
        self.o_proj2 = nn.Linear(midc, midc, 1)

        self.kln = nn.LayerNorm(self.headc)
        self.vln = nn.LayerNorm(self.headc)

        self.act = nn.GELU()
    
    def forward(self, query, key, value, name='0'):
        # B, C, H, W = query.shape
        B, L, C = query.shape
        bias = query

        q = self.q_proj(query).reshape(B, L, self.heads, self.headc).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, L, self.heads, self.headc).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, L, self.heads, self.headc).permute(0, 2, 1, 3)

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / (L)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, L, C)

        ret = v + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias


class SRNO(nn.Module):
    def __init__(self, encoder_spec, width=256, blocks=16):
        super().__init__()
        self.width = width
        self.encoder = models.make(encoder_spec)
        self.conv00 = nn.Conv2d((64 + 2)*4+2, self.width, 1)

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)
        
        
    def query_rgb(self, feat, inp, coord, cell):      
        grid = 0

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
         
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)

        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)

        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        

        ret = ret + F.grid_sample(inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        return ret

    def forward(self, feat, inp, coord, cell):
        bs, h, w, _ = coord.shape
        self.feat_l, self.feat_r = torch.chunk(rearrange(feat, '(b t) c h w -> b (t c) h w', t=2), 2, dim=1)
        self.inp_l, self.inp_r = torch.chunk(rearrange(inp, '(b t) c h w -> b (t c) h w', t=2), 2, dim=1)

        ret_l = self.query_rgb(self.feat_l, self.inp_l, coord, cell)
        ret_r = self.query_rgb(self.feat_r, self.inp_r, coord, cell)
        ret = torch.cat((ret_l, ret_r), dim=0)  # (b*2, c, h, w)

        return ret

    
class Downsample(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class ImplicitWarpModule(nn.Module):
    """ Implicit Warp Module.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 pe_wrp=True,
                 pe_x=True,
                 pe_dim = 48,
                 pe_temp = 10000,
                 warp_padding='duplicate',
                 num_heads=8,
                 aux_loss_out = False,
                 aux_loss_dim = 3,
                 window_size=2,
                 qkv_bias=True,
                 qk_scale=None,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.dim = dim
        self.pe_wrp = pe_wrp
        self.pe_x = pe_x
        self.pe_dim = pe_dim
        self.pe_temp = pe_temp
        self.aux_loss_out = aux_loss_out

        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.window_size = (window_size, window_size)
        self.warp_padding = warp_padding
        self.q = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.k = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.offset_mlp = nn.Conv2d(pe_dim * 2, 1, 3, padding=1)
        if self.aux_loss_out:
            self.proj = nn.Linear(dim, aux_loss_dim)

        self.softmax = nn.Softmax(dim=-1)
        
        self.register_parameter("position_bias", nn.Parameter(self.get_sine_position_encoding(self.window_size, pe_dim // 2, temperature=self.pe_temp, normalize=True), requires_grad=False))
        grid_h, grid_w = torch.meshgrid(
            torch.arange(0, self.window_size[0], dtype=float),
            torch.arange(0, self.window_size[1], dtype=float))

        self.num_values = self.window_size[0]*self.window_size[1]
    
        
        self.register_parameter("window_idx_offset", nn.Parameter(torch.stack((grid_h, grid_w), 2).reshape(self.num_values, 2), requires_grad=False))

    def gather_hw(self, x, idx1, idx2):
        # Linearize the last two dims and index in a contiguous x
        x = x.contiguous()
        lin_idx = idx2 + x.size(-1) * idx1
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        return x.gather(-1, lin_idx.unsqueeze(1).repeat(1,x.size(1),1).long())


    def forward(self, y, x):
        # y: frame to be propagated.
        # x: frame propagated to.
        # flow: optical flow from x to y 
        # y -> x
        n, c, h, w = x.size()
        # create mesh grid
        device = x.device
        offsets = self.offset_mlp(torch.cat([y, x], dim=1)).permute(0, 2, 3, 1) # [-1, 1]
        offsets = offsets * 2 / w
        flow = torch.cat([offsets, torch.zeros_like(offsets).to(offsets.device)], dim=-1)
        grid_h, grid_w = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
        grid = torch.stack((grid_h, grid_w), 2).repeat(n, 1, 1, 1)  # h, w, 2
        grid.requires_grad = False

        grid_wrp = grid + flow.flip(dims=(-1,)) # grid_wrp

        grid_wrp_flr = torch.floor(grid_wrp)
        grid_wrp_off = grid_wrp - grid_wrp_flr

        grid_wrp_flr = grid_wrp_flr.reshape(n, h*w, 2)
        grid_wrp_off = grid_wrp_off.reshape(n, h*w, 2)

        # get sliced windows
        grid_wrp = grid_wrp_flr.unsqueeze(2).repeat(1, 1, self.num_values, 1) + self.window_idx_offset # get 4x4 windows for each location.
        grid_wrp = grid_wrp.reshape(n, h*w*self.num_values, 2)
        if self.warp_padding == 'duplicate':
            idx0 = grid_wrp[:,:,0].clamp(min=0, max=h-1)
            idx1 = grid_wrp[:,:,1].clamp(min=0, max=w-1)
            wrp = self.gather_hw(y, idx0, idx1).reshape(n, c, h*w, self.num_values).permute(0,2,3,1).reshape(n, h*w*self.num_values, c)
        elif self.warp_padding == 'zero':
            invalid_h = torch.logical_or(grid_wrp[:,:,0]<0, grid_wrp[:,:,0]>h-1)
            invalid_w = torch.logical_or(grid_wrp[:,:,1]<0, grid_wrp[:,:,1]>h-1)
            invalid = torch.logical_or(invalid_h, invalid_w)

            idx0 = grid_wrp[:,:,0].clamp(min=0, max=h-1)
            idx1 = grid_wrp[:,:,1].clamp(min=0, max=w-1)

            wrp = self.gather_hw(y, idx0, idx1).reshape(n, c, h*w, self.num_values).permute(0,2,3,1).reshape(n, h*w*self.num_values, c)
            wrp[invalid] = 0
        else:
            raise ValueError(f'self.warp_padding: {self.warp_padding}')
        
        # add sin/cos positional encoding to 4x4 windows
        wrp_pe = self.position_bias.repeat(n, h*w, 1)

        if self.pe_wrp:
            wrp = wrp.repeat(1,1,self.pe_dim//c) + wrp_pe
        else:
            wrp = wrp.repeat(1,1,self.pe_dim//c)

        # add postional encoding to source pixel
        x = x.flatten(2).permute(0,2,1)
        x_pe = self.get_sine_position_encoding_points(grid_wrp_off, self.pe_dim // 2, temperature=self.pe_temp, normalize=True)

        if self.pe_x:
            x = x.repeat(1,1,self.pe_dim//c) + x_pe
        else:
            x = x.repeat(1,1,self.pe_dim//c)

        nhw = n*h*w
        
        kw = self.k(wrp).reshape(nhw, self.num_values, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) 
        vw = self.v(wrp).reshape(nhw, self.num_values, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        qx = self.q(x).reshape(nhw, self.num_heads, self.dim // self.num_heads).unsqueeze(1).permute(0, 2, 1, 3)


        attn = (qx * self.scale) @ kw.transpose(-2, -1)
        attn = self.softmax(attn)
        out = (attn @ vw).transpose(1, 2).reshape(nhw, 1, self.dim)

        out = out.squeeze(1)

        if self.aux_loss_out:
            out_rgb = self.proj(out).reshape(n, h, w, c).permute(0,3,1,2)
            return out.reshape(n, h, w, self.dim).permute(0,3,1,2), out_rgb
        else:
            return out.reshape(n, h, w, self.dim).permute(0,3,1,2)

    def get_sine_position_encoding_points(self, points, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        """ get_sine_position_encoding_points for single points.

        Args:
            points (tuple[int]): The temporal length, height and width of the window.
            num_pos_feats
            temperature
            normalize
            scale
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            mut_attn (bool): If True, add mutual attention to the module. Default: True
        """

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi


        y_embed, x_embed = points[:,:,0].unsqueeze(0), points[:,:, 1].unsqueeze(0)
        
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (self.window_size[0] + eps) * scale
            x_embed = x_embed / (self.window_size[1] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device='cuda')
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3)

        return pos_embed.squeeze(0)

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        """ Get sine position encoding """
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32) - 1
        x_embed = not_mask.cumsum(2, dtype=torch.float32) - 1
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()
        