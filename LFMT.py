import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from einops import rearrange, repeat
from typing import Optional, Callable, Any
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from functools import partial
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.vision_transformer import _load_weights

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),  
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)    #64// squeeze_factor=30???需要调整？
        )

    def forward(self, x):
        return self.cab(x)

#SSMB
class SSMB(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path  
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        ) #？？？
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,  
            kernel_size=d_conv,
            groups=self.d_inner//2, 
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float()) 
        
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2)) #对输入的操作
        
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen) 
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1) 
        y = rearrange(y, "b d l -> b l d") 
        out = self.out_proj(y)
        return out

#Residue Subspace Mamba Block
class RSMB(nn.Module): 
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,   
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  
        self.self_attention = SSMB(d_model=hidden_dim, d_state=d_state,expand=expand, ) 
        self.drop_path = DropPath(drop_path)     
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim,is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # 输入x [B,HW,C]   输出x [B,HW,C] 
        B, L, C = input.shape
        # input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))  
        x = x.view(B, *x_size, C).contiguous()
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() #进行通道注意力，输入为[B,C,H,W]
        x = x.view(B, -1, C).contiguous()
        return x

#To build the module    SA-RSMB   EPMB
class BasicLayer(nn.Module):
   
    def __init__(self,
                 dim,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(RSMB(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,))
            
    def forward(self, x, x_size):
        # x: [b, l, c]
        h, w = x_size
        x = rearrange(x, 'b c h w -> b (h w) c')
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class ResidualGroup(nn.Module):
   
    def __init__(self,
                 angRes,
                 dim,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 resi_connection='1conv', 
                ):
        super(ResidualGroup, self).__init__()
        self.angRes = angRes
        self.dim = dim
        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
       
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
            
    def forward(self, x):

        b, c, h, w = x.shape
        x_size = [h, w]
        x = self.conv(self.residual_group(x, x_size)) + x

        return x

class SpaSSM(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.layer = ResidualGroup(angRes=angRes, dim=dim, depth=depth)

    def forward(self, x):
        x = rearrange(x, 'b c a h w -> (b a) c h w')
        x = self.layer(x)
        x = rearrange(x, '(b a) c h w -> b c a h w', a=self.angRes ** 2)
        return x

class AngSSM(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.layer = ResidualGroup(angRes=angRes, dim=dim,  depth=depth)  #input_resolution=(5, 5),

    def forward(self, x):
        x = rearrange(x, 'b c (u v) h w -> (b h w) c u v', u=self.angRes)
        x = self.layer(x)
        x = rearrange(x, '(b h w) c u v -> b c (u v) h w', h=self.h, w=self.w)
        return x

#SA-RSMB
class SpaAngFilter(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.spa_block = SpaSSM(angRes=angRes, dim=dim, depth=depth)
        self.ang_block = AngSSM(angRes=angRes, dim=dim, depth=depth)

    def forward(self, x):
        x = self.spa_block(x)
        x = self.ang_block(x)
        return x

#EPMB
class EPISSM(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.layer = ResidualGroup(angRes=angRes, dim=dim,  depth=depth)  #Weight parameter sharing

    def forward(self, x):
        x = rearrange(x, 'b c (u v) h w -> (b v w) c u h', u=self.angRes)
        x = self.layer(x)
        x = rearrange(x, '(b v w) c u h -> (b u h) c v w', v=self.angRes, w=self.w)
        x = self.layer(x)
        x = rearrange(x, '(b u h) c v w -> b c (u v) h w', u=self.angRes, h=self.h)
        return x


#EPTB
class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.epi_trans = BasicTrans(channels, channels*2)
        # self.conv = nn.Sequential(
        #     nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        # )
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )
        
    def forward(self, buffer):
        shortcut1 = buffer
        [_, _, _, h, w] = buffer.size()
        self.epi_trans.mask_field = [self.angRes * 2, 11]  
        # Horizontal
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes) 
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut1 

        # Vertical
        shortcut2 = buffer
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut2 

        return buffer

#Simple Transformer Block  STB
class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False) 
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )  
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)  

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device) #self.mask_field[0]？

        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')  #N B C
        epi_token = self.linear_in(epi_token)  

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token   

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer


class Net(nn.Module):
    def __init__(self, angRes, scale_factor, channel):
        super(Net, self).__init__()
        self.channels = channel
        self.angRes = angRes
        self.scale = scale_factor
        # depth = 2
        
        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Non-local Spatial-Angular Correlation Learning #############
        self.altblock = nn.Sequential(
            SpaAngFilter(dim=channel, angRes=self.angRes, depth=2),
            SpaAngFilter(dim=channel, angRes=self.angRes, depth=2),
            SpaAngFilter(dim=channel, angRes=self.angRes, depth=2),
        )
        
        ############# Non-local Spatial-Angular Correlation Enhancement #############
        self.epiblock1 = nn.Sequential(
            EPISSM(dim=channel, angRes=self.angRes, depth=1),
            EPISSM(dim=channel, angRes=self.angRes, depth=1),
            EPISSM(dim=channel, angRes=self.angRes, depth=1),
        ) 
        self.epiblock2 = nn.Sequential(
            AltFilter(self.angRes, channel),
            AltFilter(self.angRes, channel),
            AltFilter(self.angRes, channel),
        )
        
        ########################### Hierarchical Feature Fusion and Upsampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channel * 3, channel * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr, info=None,return_features=False):
        lr = rearrange(lr, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)
        [b, c, u, v, h, w] = lr.size()

        sr_y = LF_interpolate(lr, scale_factor=self.scale, mode='bicubic')
        sr_y = rearrange(sr_y, 'b c u v h w -> b c (u h) (v w)', u=u, v=v)

        # Initial Feature Extraction  
        x = rearrange(lr, 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer_init = self.conv_init(buffer) + buffer  

        for m in self.modules():
            m.h = lr.size(-2)
            m.w = lr.size(-1) 

        # Non-local Spatial-Angular Correlation Modeling
        
        buffer_SA = self.altblock(buffer_init) + buffer_init  #输入 b c (u v) h w
        buffer_EPIM = self.epiblock1(buffer_SA) + buffer_SA
        buffer_EPIT = self.epiblock2(buffer_SA) + buffer_SA
        
        # UP-Sampling
        buffer = torch.cat([buffer_init, buffer_EPIM, buffer_EPIT], dim=1) # 无buffer_SA
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=u, v=v) 
        y = self.upsampling(buffer) + sr_y
        upsample_partial = buffer
        for i in range(1):
           upsample_partial = self.upsampling[i](upsample_partial)
        upsample_partial = rearrange(upsample_partial, 'b c (u h) (v w) -> b c (u v) h w', u=u, v=v)  #融合后的特征
        
        if return_features:
           return y, buffer_EPIM, buffer_EPIT, upsample_partial
        else:
           return y

def LF_interpolate(LF, scale_factor, mode):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_upscale

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out, HR)

        return loss


def weights_init(m):
    pass

if __name__ == '__main__':
    torch.cuda.set_device("cuda:1")
    net = Net(5, 4, 64).cuda()   
    print(net)
    from thop import profile
    input = torch.randn(1, 1, 160, 160).cuda() #32*5=160
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
