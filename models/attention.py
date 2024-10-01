import copy
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import torch.nn.functional as F

from models.submodule import convbn

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class AttentionFuse(nn.Module):
    def __init__(self, hidden_dim, nhead, num_attn_layers, isContext=False, ksize=3, stride=1, pad=1, dilation=1):
        super().__init__()

        if isContext:
            context_layer = ContextLayer(hidden_dim, ksize, stride, pad, dilation)
            self.context_layers = get_clones(context_layer, num_attn_layers)

        attn_layer = AttentionLayer(hidden_dim, nhead)
        self.attn_layers = get_clones(attn_layer, num_attn_layers)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers
        self.isContext = isContext

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor):
        b, c, h, w = feat_left.shape

        # reshape
        feat_left = feat_left.permute(0, 2, 3, 1).reshape(b*h, w, c)  # BxHxWxC -> BHxWxC
        feat_right = feat_right.permute(0, 2, 3, 1).reshape(b*h, w, c)  # BxHxWxC -> BHxWxC

        # update
        for idx in range(self.num_attn_layers):
            if self.isContext:
                feat_left, feat_right = self.context_layers[idx](feat_left, feat_right, b, h)
            # checkpoint attn
            def create_attn(module):
                def attn(*inputs):
                    return module(*inputs)
                return attn
            # feat_left, feat_right = checkpoint(create_attn(self.attn_layers[idx]), feat_left, feat_right)
            feat_left, feat_right = self.attn_layers[idx](feat_left, feat_right)

        # reshape
        feat_left = feat_left.view(b, h, w, c).permute(0, 3, 1, 2)
        feat_right = feat_right.view(b, h, w, c).permute(0, 3, 1, 2)

        return feat_left, feat_right


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()

        self.embed_dim = hidden_dim
        self.num_heads = nhead

        factory_kwargs = {'device': None, 'dtype': None}

        self.l_proj_weight = Parameter(torch.empty((2 * self.embed_dim, self.embed_dim), **factory_kwargs))
        self.r_proj_weight = Parameter(torch.empty((2 * self.embed_dim, self.embed_dim), **factory_kwargs))

        self.l_proj_bias = Parameter(torch.empty(2 * self.embed_dim, **factory_kwargs))
        self.r_proj_bias = Parameter(torch.empty(2 * self.embed_dim, **factory_kwargs))

        self.l_out_proj = NonDynamicallyQuantizableLinear(self.embed_dim, self.embed_dim, bias=True,
                                                          **factory_kwargs)
        self.r_out_proj = NonDynamicallyQuantizableLinear(self.embed_dim, self.embed_dim, bias=True,
                                                          **factory_kwargs)

        self.norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.l_proj_weight)
        xavier_uniform_(self.r_proj_weight)
        constant_(self.l_proj_bias, 0.)
        constant_(self.r_proj_bias, 0.)
        constant_(self.l_out_proj.bias, 0.)
        constant_(self.r_out_proj.bias, 0.)

    def forward(self, feat_left: Tensor, feat_right: Tensor):

        bh, w, c = feat_left.size()
        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim

        # norm
        l_k = self.norm(feat_left)
        r_k = self.norm(feat_right)

        # [BH,W,C]  [C,C] -> BH,W,C
        l_k, l_v = F.linear(l_k, self.l_proj_weight, self.l_proj_bias).chunk(2, dim=-1)
        r_k, r_v = F.linear(r_k, self.r_proj_weight, self.r_proj_bias).chunk(2, dim=-1)

        l_k = l_k.reshape(bh, w, self.num_heads, head_dim)  # BH,W,E,dim
        r_k = r_k.reshape(bh, w, self.num_heads, head_dim)  # BH,W,E,dim
        l_v = l_v.reshape(bh, w, self.num_heads, head_dim)  # BH,W,E,dim
        r_v = r_v.reshape(bh, w, self.num_heads, head_dim)  # BH,W,E,dim

        # get attention weight
        attn_weight = torch.einsum('nwec,nvec->newv', l_k, r_k) * 0.1  # BHxExWxW'

        # update right value
        attn_sm_l = F.softmax(attn_weight, dim=-1)
        attn_sm_l = attn_sm_l.reshape(bh*self.num_heads, w, w)
        r_v = r_v.transpose(1, 2).reshape(bh*self.num_heads, w, head_dim)
        r_v = torch.bmm(attn_sm_l, r_v)  # [BHE,W,W]  [BHE,W,dim] -> [BHE,W,dim]
        r_v = r_v.reshape(bh, self.num_heads, w, head_dim).transpose(1, 2).reshape(bh, w, c)

        # update left value
        attn_sm_r = attn_weight.transpose(-1, -2)
        attn_sm_r = F.softmax(attn_sm_r, dim=-1)
        attn_sm_r = attn_sm_r.reshape(bh*self.num_heads, w, w)
        l_v = l_v.transpose(1, 2).reshape(bh*self.num_heads, w, head_dim)
        l_v = torch.bmm(attn_sm_r, l_v)  # [BHE,W,W]  [BHE,W,dim] -> BHE,W,dim
        l_v = l_v.reshape(bh, self.num_heads, w, head_dim).transpose(1, 2).reshape(bh, w, c)

        # linear
        r_v = F.linear(r_v, self.l_out_proj.weight, self.l_out_proj.bias)
        l_v = F.linear(l_v, self.r_out_proj.weight, self.r_out_proj.bias)

        # add
        feat_left = feat_left + r_v
        feat_right = feat_right + l_v

        return feat_left, feat_right


class ContextLayer(nn.Module):
    def __init__(self, hidden_dim, ksize, stride, pad, dilation):
        super().__init__()

        self.conv = convbn(hidden_dim, hidden_dim, ksize, stride, pad, dilation)

    def forward(self, feat_left: Tensor, feat_right: Tensor, b: int, h: int):
        _, w, c = feat_left.shape

        # reshape, [B, C, H, W]
        feat_left_2 = feat_left.reshape(b, h, w, c).permute(0, 3, 1, 2)
        feat_right_2 = feat_right.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # cnn
        feat_left_2 = self.conv(feat_left_2)
        feat_right_2 = self.conv(feat_right_2)

        # reshape
        feat_left_2 = feat_left_2.permute(0, 2, 3, 1).reshape(b*h, w, c)
        feat_right_2 = feat_right_2.permute(0, 2, 3, 1).reshape(b*h, w, c)

        # add
        feat_left = feat_left + feat_left_2
        feat_right = feat_right + feat_right_2

        return feat_left, feat_right

