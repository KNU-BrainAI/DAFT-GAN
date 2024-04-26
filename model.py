import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

import einops
from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo

N_GROUPS = 32

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.GroupNorm(N_GROUPS, out_planes),
        nn.ReLU(inplace=False))
    return block
    
    
# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = 10
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code




##################### MaskEncoder ##########################
class MaskConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, mask, dmask, skip_mask=False):
        if skip_mask:
            feature = self.conv(x)
        else:
            full_feature = self.conv(x)
            masked_feature = self.conv2(x * mask)
            feature = full_feature * (1. - dmask) + masked_feature * dmask
        return feature



class MaskGroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        super(MaskGroupNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(1, num_channels, 1, **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(1, num_channels, 1, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

    def forward(self, x, mask=None):
        original_shape = x.shape
        N = original_shape[0]
        C = original_shape[1]
        x = x.view(N, self.num_groups, -1)
        D = x.shape[-1]

        if mask is not None:
            mask = mask.view(N, -1)
            mask = einops.repeat(mask, "b wh -> b c wh", c=C)
            
            mask = mask.reshape(N, self.num_groups, -1)
            m = einops.repeat(einops.reduce(mask, "b c d -> b c", 'sum'), "b c -> b c d", d=D)
            n = torch.ones_like(m) * D
            mean_bias = x.mean(dim=-1, keepdim=True)
            mean_real = mean_bias * n / (n - m + self.eps)
            x_fulfill = x * (1. - mask) + mean_real * mask
            var_bias = x_fulfill.var(dim=-1, keepdim=True)
            var_real = var_bias * (n - 1) / (n - m - 1 + self.eps)
        else:
            mean_real = x.mean(dim=-1, keepdim=True)
            var_real = x.var(dim=-1, keepdim=True)

        x = ((x - mean_real) / (var_real + self.eps).sqrt())
        if self.affine:
            x = x.view(N, C, -1)
            x = x * self.weight + self.bias

        return x.view(*original_shape)
    


class MaskNormalize(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = MaskGroupNorm(num_groups=N_GROUPS, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x, mask, skip_mask=False):
        if skip_mask:
            normed = self.norm(x)
        else:
            mask_c = mask
            mask = mask.squeeze(1)
            full_normed = self.norm(x)
            masked_normed = self.norm(x * mask_c, (1. - mask))

            normed = full_normed * (1. - mask_c) + masked_normed * mask_c
        return normed
    


class MaskEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                normalization=True, activation=True):
        super().__init__()

        if activation:
            self.act = Swish()
        else:
            self.act = None

        if kernel_size == 3:
            self.m_down = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        if kernel_size == 4:
            self.m_down = torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.m_conv = MaskConv2d(in_channels, out_channels, kernel_size, stride, padding=1)

        if normalization:
            self.norm = MaskNormalize(out_channels)
        else:
            self.norm = None

    def forward(self, x, mask):
    
        dmask = -self.m_down(-mask)
        if self.act is not None:
            x = self.act(x)
        x = self.m_conv(x, mask, dmask)
        if self.norm is not None:
            x = self.norm(x, dmask)

        return x, dmask


######################## LSTM #########################

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, z_dim=100):
        super(CustomLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
        self.noise2h = nn.Linear(z_dim, hidden_sz)
        self.noise2c = nn.Linear(z_dim, hidden_sz)
        self.hidden_seq = []

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self, noise):
        h_t = self.noise2h(noise)
        c_t = self.noise2c(noise)

        self.c_t = c_t
        self.h_t = h_t

    def forward(self, x):
        c_t = self.c_t
        h_t = self.h_t
        HS = self.hidden_size
        x_t = x

        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
		torch.sigmoid(gates[:, :HS]), # input
		torch.sigmoid(gates[:, HS:HS*2]), # forget
		torch.tanh(gates[:, HS*2:HS*3]),
		torch.sigmoid(gates[:, HS*3:]), # output
	)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        self.h_t = h_t
        self.c_t = c_t

        return h_t, c_t


###################### Discriminator ######################


class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)        
        return x + self.gamma*res
    

def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


class NetD(nn.Module):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetD, self).__init__()
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        in_out_pairs = get_D_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))
        
        self.COND_DNET = NetC(64)
        
    def forward(self,x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out

######################## NetC ###########################

class NetC(nn.Module):
    def __init__(self, ndf, cond_dim=256):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf*8+cond_dim, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False),
        )
    def forward(self, out, y):
        y = y.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out 


####################### Generator ##########################

class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=False)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=False)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

class CrossRefine(nn.Module):
    def __init__(self, cond_dim, word_dim, out_ch):
        super(CrossRefine, self).__init__()
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine1 = CrossAffine(cond_dim, word_dim, out_ch)
        self.affine2 = CrossAffine(cond_dim, word_dim, out_ch)
    
    def residual(self, h, y, attn_out):
        h = self.affine1(h, y, attn_out)
        h = nn.ReLU(inplace=False)(h)
        h = self.c1(h)
        h = self.affine2(h, y, attn_out)
        h = nn.ReLU(inplace=False)(h)
        h = self.c2(h)
        return h
    
    def forward(self, x, text, attn_out):
        x = x + self.residual(x, text, attn_out)
        return x
    
class CrossAffine(nn.Module):
    def __init__(self, cond_dim, word_dim, num_features):
        super().__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim+word_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=False)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim+word_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=False)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))
        self._initialize()
    
    def _initialize(self):
        nn.init.kaiming_uniform_(self.fc_gamma.linear2.weight.data, nonlinearity='relu')
        nn.init.zeros_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)        
    
    def forward(self, x, text, attn_out):
        t = text.unsqueeze(2).unsqueeze(3)
        t = t.expand(-1, -1, x.size(2), x.size(3))
        
        code = torch.cat([attn_out, t], dim=1).permute(0, 2, 3, 1) # batch x ih x iw x cdf
        r = self.fc_gamma(code).permute(0, 3, 1, 2) # batch x num_features x ih x iw
        B = self.fc_beta(code).permute(0, 3, 1, 2)
        return r * x + B
    

class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, lstm, upsample=False, vis_refine=False):
        super(G_Block, self).__init__()
        self.lstm = lstm
        self.upsample = upsample
        self.vis_refine = vis_refine
        self.affine0 = Affine(cond_dim, out_ch)
        self.affine1 = Affine(cond_dim, out_ch)
        self.affine2 = Affine(cond_dim, out_ch)
        self.affine3 = Affine(cond_dim, out_ch)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        if self.upsample:
            self.upconv = upBlock(in_ch, out_ch)

        if vis_refine:
            self.att = ATT_NET(out_ch, 256)
            self.refine_block = CrossRefine(cond_dim, 256, out_ch)
            self.channel_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, h, yy):
        lstm_input = yy
        y, _ = self.lstm(lstm_input)
        h = self.affine0(h, y)
        h = nn.LeakyReLU(0.2, inplace=False)(h)

        lstm_input = yy
        y, _ = self.lstm(lstm_input)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=False)(h)

        h = self.c1(h)

        lstm_input = yy
        y, _ = self.lstm(lstm_input)
        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2, inplace=False)(h)

        lstm_input = yy
        y, _ = self.lstm(lstm_input)
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2, inplace=False)(h)
        
        h = self.c2(h)
        return h, y

    def forward(self, x, text, enc_x, words_embs, text_mask, pre_vis=None):
        if self.upsample==True:
            x = self.upconv(x)
        enc_x = torch.cat(enc_x, dim=1) if type(enc_x) is tuple else enc_x
        x = x + enc_x

        res_out, y = self.residual(x, text)
        out = x + res_out

        if self.vis_refine:
            self.att.applyMask(text_mask)
            attn_out, attn = self.att(out, words_embs)
            vis_out = self.refine_block(out, y, attn_out)
            if pre_vis is not None:
                pre_vis = F.interpolate(pre_vis, scale_factor=2, mode='bilinear', align_corners=True)
                vis_out = self.channel_conv(pre_vis) + vis_out
            return vis_out, out, attn
        else:
            return out


class NetG(nn.Module):
    def __init__(self, c_img=3, cnum=64, z_dim=100, cond_dim=256):
        super().__init__()
        self.cnum = cnum
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        self.lstm = CustomLSTM(cond_dim, cond_dim)

        self.en_1 = MaskEncoderBlock(c_img, cnum, 3, 1, normalization=False, activation=False)
        self.en_2 = MaskEncoderBlock(cnum, cnum*2, 4, 2)
        self.en_3 = MaskEncoderBlock(cnum*2, cnum*4, 4, 2)
        self.en_4 = MaskEncoderBlock(cnum*4, cnum*8, 4, 2)
        self.en_5 = MaskEncoderBlock(cnum*8, cnum*8, 4, 2)
        self.en_6 = MaskEncoderBlock(cnum*8, cnum*8, 4, 2)
        self.en_7 = MaskEncoderBlock(cnum*8, cnum*8, 4, 2)

        self.fc = nn.Linear(z_dim, cnum*8*4*4)
        
        self.de_7 = G_Block(cond_dim, cnum*8, cnum*8, self.lstm, upsample=False)
        self.de_6 = G_Block(cond_dim, cnum*8, cnum*8, self.lstm, upsample=True)
        self.de_5 = G_Block(cond_dim, cnum*8, cnum*8, self.lstm, upsample=True, vis_refine=True)
        self.de_4 = G_Block(cond_dim, cnum*8, cnum*8, self.lstm, upsample=True, vis_refine=True)
        self.de_3 = G_Block(cond_dim, cnum*8, cnum*4, self.lstm, upsample=True, vis_refine=True)
        self.de_2 = G_Block(cond_dim, cnum*4, cnum*2, self.lstm, upsample=True, vis_refine=True)
        self.de_1 = G_Block(cond_dim, cnum*2, cnum, self.lstm, upsample=True, vis_refine=True)
        self.to_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(cnum, c_img, 3, 1, 1),
            nn.Tanh(),
            )
        
        self.weight_conv = nn.Conv2d(16, 1, 3, 1, padding=1)
    
    def forward(self, x, mask, z_code, sent_emb, words_embs, text_mask):
        # hidden feature initialization
        self.lstm.init_hidden(z_code)
        
        ################## Image Encoder ###################
        out_1, dmask_1 = self.en_1(x, mask)     # batch x 64 x 256 x 256
        out_2, dmask_2 = self.en_2(out_1, dmask_1) # batch x 128 x 128 x 128
        out_3, dmask_3 = self.en_3(out_2, dmask_2) # batch x 256 x 64 x 64
        out_4, dmask_4 = self.en_4(out_3, dmask_3) # batch x 512 x 32 x 32
        out_5, dmask_5 = self.en_5(out_4, dmask_4) # batch x 512 x 16 x 16
        out_6, dmask_6 = self.en_6(out_5, dmask_5) # batch x 512 x 8 x 8
        out_7, dmask_7 = self.en_7(out_6, dmask_6) # batch x 512 x 4 x 4
        
        h_code = self.fc(z_code)
        h_code = h_code.view(z_code.size(0), 8*self.cnum, 4, 4)
        
        ##################### Decoder #########################
        dout_4 = self.de_7(h_code, sent_emb, out_7, words_embs, text_mask) # batch x 512 x 4 x 4
        
        dout_8 = self.de_6(dout_4, sent_emb, out_6, words_embs, text_mask) # batch x 512 x 8 x 8

        v_dout_16, dout_16, att_16 = self.de_5(dout_8, sent_emb, out_5, words_embs, text_mask) # batch x 512 x 16 x 16

        v_dout_32, dout_32, att_32 = self.de_4(dout_16, sent_emb, out_4, words_embs, text_mask, pre_vis=v_dout_16) # batch x 512 x 32 x 32

        v_dout_64, dout_64, att_64 = self.de_3(dout_32, sent_emb, out_3, words_embs, text_mask, pre_vis=v_dout_32) # batch x 256 x 64 x 64

        v_dout_128, dout_128, att_128 = self.de_2(dout_64, sent_emb, out_2, words_embs, text_mask, pre_vis=v_dout_64) # batch x 128 x 128 x 128

        v_dout_256, dout_256, att_256 = self.de_1(dout_128, sent_emb, out_1, words_embs, text_mask, pre_vis=v_dout_128) # batch x 64 x 256 x 256

        dout = self.to_img(v_dout_256) # batch x 3 x 256 x 256
        
        ###################### Attention Loss ######################
        
        out = x * (1. - mask) + dout * mask
        
        attn_loss_add = torch.zeros(att_256.size(0), 16, att_256.size(2), att_256.size(3), device=att_256.device)
        att = torch.cat([att_256, attn_loss_add], 1)
        att = att[:, :16, :, :]
        att = self.weight_conv(att)
        return out, dout, att

