"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack, repeat, reduce
from functools import partial


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.proj = nn.Linear(idf, cdf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x cdf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        targetT = self.proj(targetT) 
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context
        # --> batch x cdf x sourceL
        # Get attention
        # (batch x queryL x cdf)(batch x cdf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x cdf x sourceL)(batch x sourceL x queryL)
        # --> batch x cdf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)
        return weightedContext, attn

class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim = 1)
        return normed * self.scale * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma


class ATT_Block(nn.Module):
    def __init__(self, idf, cdf, ff_mul=2):
        super().__init__()
        self.attn = GlobalAttentionGeneral(idf, cdf)
        self.ff = nn.Sequential(
            ChannelRMSNorm(idf),
            nn.Conv2d(idf, idf*ff_mul, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(idf*ff_mul, idf, kernel_size=1, padding=0)
        )
    
    def forward(self, x, context, text_mask):
        self.attn.applyMask(text_mask)
        attn_x, attn = self.attn(x, context)
        x = attn_x + x
        x = self.ff(x) + x
        return attn_x, attn

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dot_product = True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.dot_product = dot_product

        self.norm = ChannelRMSNorm(dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_inner, 1, bias = False) if dot_product else None
        self.to_v = nn.Conv2d(dim, dim_inner, 1, bias = False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)

    def forward(self, fmap):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = fmap.shape[0]

        fmap = self.norm(fmap)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, v = self.to_q(fmap), self.to_v(fmap)

        k = self.to_k(fmap) if self.to_k else q

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = self.heads), (q, k, v))

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # l2 distance or dot product

        if self.dot_product:
            sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            # using pytorch cdist leads to nans in lightweight gan training framework, at least
            q_squared = (q * q).sum(dim = -1)
            k_squared = (k * k).sum(dim = -1)
            l2dist_squared = rearrange(q_squared, 'b i -> b i 1') + rearrange(k_squared, 'b j -> b 1 j') - 2 * einsum('b i d, b j d -> b i j', q, k) # hope i'm mathing right
            sim = -l2dist_squared

        # scale

        sim = sim * self.scale

        # attention

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, dim_context, dim_head=64, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads
        kv_input_dim = dim_context
        
        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(kv_input_dim)
        
        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_kv = nn.Linear(kv_input_dim, dim_inner * 2, bias=False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)
    
    def forward(self, fmap, context, mask=None):
        fmap = self.norm(fmap)
        context = self.norm_context(context)
        
        x, y = fmap.shape[-2:]
        
        h = self.heads
        
        q, k, v = (self.to_q(fmap), *self.to_kv(context).chunk(2, dim = -1)) # q : 4 x 256 x 2 x 2
        
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (k, v)) # 16 x 16 x 64
        
        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h = self.heads) # 16 x 4 x 64
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # (4 * 4) x 4 x 16
        
        mask = repeat(mask, 'b j -> (b h) 1 j', h = self.heads)
        sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim = -1) # (B x N) x queryL x sourceL

        out = einsum('b i j, b j d -> b i d', attn, v) # (4 * 4) x 4 x 64
        
        attn = rearrange(attn, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)
        return self.to_out(out), attn
    

def FeedForward(
    dim,
    mult = 4,
    channel_first = False
):
    dim_hidden = int(dim * mult)
    norm_klass = ChannelRMSNorm if channel_first else RMSNorm
    proj = partial(nn.Conv2d, kernel_size = 1) if channel_first else nn.Linear

    return nn.Sequential(
        norm_klass(dim),
        proj(dim, dim_hidden),
        nn.GELU(),
        proj(dim_hidden, dim)
    )
    
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        dot_product = False
    ):
        super().__init__()
        self.attn = SelfAttention(dim = dim, dim_head = dim_head, heads = heads, dot_product = dot_product)
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = CrossAttention(dim = dim, dim_context = dim_context, dim_head = dim_head, heads = heads)
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)

    def forward(self, x, context, mask = None):
        attn, att_map = self.attn(x, context=context, mask = mask)
        x = attn + x
        x = self.ff(x) + x
        return x, att_map


