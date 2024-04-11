import torch
import torch.nn as nn

from mmcv.cnn import normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmaction.models.builder import BACKBONES
from mmaction.utils import get_root_logger

cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

"""

8 heads, 4 transformer encoder layers
"""

"""
    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 2, 1, 3).contiguous()  # N M T C V
        # keep the rest of the dim the same, operates on each joint separately
        x = x.permute(0, 4, 2, 1, 3).contiguous().view(-1, v)
        y = self.dense(x)
"""


class AddPositionEmbedding(nn.Module):
    def __init__(self, shape=(1, 21, 32, 252)):
        super().__init__()
        self.position = nn.Parameter(torch.rand(shape), requires_grad=True)

    def forward(self, inputs):
        return inputs + torch.tensor(self.position, dtype=inputs.dtype)


def pos_embed(input, d_model):
    input = input.view(-1, 1).to(cuda_device)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))  # 21,dim
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


class TemporalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        # self.joints = joints  # 21

    def forward(self, inputs):
        """input: b,t,v,c
        should be reshapped to b,V,T,C = 64,21,32,252
        sinusoid encoding table num_frames, channels  = max_len,d_model
        temporal pos embedding
        pos = torch.arange(max_len, dtype=torch.float32)
        out = pos_embed(pos, d_model) T,V
        """
        pos = torch.arange(self.max_len, dtype=torch.float32)
        out = pos_embed(pos, self.d_model)

        return out


class NodeEmbedding(nn.Module):
    def __init__(self, input_units, out_units):
        super().__init__()

        """
        a nivel de nodo 1x3 -> 
        """
        self.input_units = input_units
        self.out_units = out_units  # out_joints = dim =252
        self.expand_dim = out_units // 2  # 252//2 = 126
        # self.position = nn.Parameter(torch.rand((1, 21, 252)), requires_grad=True)
        self.position = pos_embed(torch.arange(21, dtype=torch.float32), self.out_units)  # 21 joints

        self.dense = nn.Sequential(
            *[nn.Linear(self.input_units, 63),  # x2
              nn.ReLU(),
              nn.Linear(63, self.expand_dim),  # 126
              nn.ReLU(),
              nn.Linear(self.expand_dim, self.out_units)])  # 252

        self.dense.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)

    def forward(self, x):
        # return the embedding of the landmarks
        n, t, v, c = x.size()
        x = x.view(-1, v, c)  # embedding de cada nodo
        y = self.dense(x)
        # todo: add node position here
        y = y + self.position
        y = y.view(n, t, v, -1)
        # 64,252,32,21 -> N, C, T, V
        return y


class TemporalMHAttention(nn.Module):
    def __init__(self, dim, h, dropout=0.0):
        super().__init__()

        self.dim = dim  # hidden dim , input embedding dim
        self.h = h  # num_heads
        self.scale = self.dim ** -0.5
        self.embed_dim = 3 * dim  # 252*3= 756
        self.q = nn.Linear(dim, self.embed_dim, bias=False)
        self.k = nn.Linear(dim, self.embed_dim, bias=False)
        self.v = nn.Linear(dim, self.embed_dim, bias=False)
        self.drop1 = nn.Dropout(dropout)
        self.proj = nn.Linear(self.embed_dim, dim, bias=False)

    def forward(self, inputs):
        # nTVC -> nVTC
        # x = inputs.permute(0, 2, 1, 3).contiguous()  # N V T C
        x = inputs
        b, v, t, c = x.size()
        x = x.view(-1, t, c)
        query = self.q(x)  # 64,32,756
        key = self.k(x)  # 64,32,756
        value = self.v(x)  # 64,32,756
        # head_dim = 756//6= 126

        query = query.view((b * v, t, self.h, self.embed_dim // self.h)).permute(0, 2, 1, 3)  # b, h, t,head_dim
        key = key.view((b * v, t, self.h, self.embed_dim // self.h)).permute(0, 2, 1, 3)
        value = value.view((b * v, t, self.h, self.embed_dim // self.h)).permute(0, 2, 1, 3)

        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # b, h, t, t
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop1(attn)

        out = torch.matmul(attn, value)  # b,h,t,dv
        # out = out.permute(0, 2, 1, 3).contiguous().view(b, t, self.h * self.dim)  # to b,t,h,dv -> b,t,h*dv
        out = out.permute(0, 2, 1, 3).contiguous().view(b * v, t, self.embed_dim)  # to b,t,h,dv -> b,t,h*dv
        out = self.proj(out)  # b*v, t, dim
        out = out.view(b, v, t, c)
        return out


class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads, expand, attn_dropout=0.2, drop_rate=0.2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.expand = expand

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate)

        self.attn_dropout = attn_dropout
        self.attention = TemporalMHAttention(dim=dim, h=heads, dropout=attn_dropout)

        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.GELU()
        self.num_frames = 32
        self.joints = 21
        self.norm = nn.LayerNorm([self.joints, self.num_frames, dim])  # vtc
        # MLP
        self.expand_dense = nn.Linear(dim, dim * expand, bias=False)  # dim=252 -> out_dim = 252*2
        self.dense = nn.Linear(dim * expand, dim, bias=False)  # 252*2> out_dim = 252

    def forward(self, inputs):
        x = inputs
        n, t, v, c = x.size()
        # LayerNorm
        x = self.norm(x)
        # MHA
        x = self.attention(x)
        x = self.dropout(x)  # applied to the normalized outputs from the MHA layer
        x = inputs + x  # element wise adding operation
        attn_out = x
        # LayerNorm
        x = self.norm(x)
        # MLP o FFN
        x = self.expand_dense(x)  # FC dim*4
        x = self.activation(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = attn_out + x

        return x


@BACKBONES.register_module()
class JointsFormer(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 max_len,
                 input_dim,
                 output_dim,
                 pretrained=None):
        super().__init__()

        self.pretrained = pretrained
        self.dim = dim
        self.heads = heads
        # todo: node embedding + npos
        self.embedding = NodeEmbedding(input_dim, output_dim)
        # todo: add temporal embedding
        self.joints = 21
        # self.temp_pos = TemporalEmbedding(max_len=max_len, d_model=dim)  # joints, 32,252
        self.temp_pos = AddPositionEmbedding()
        self.transformer1 = TemporalTransformer(self.dim, self.heads, expand=2)
        self.transformer2 = TemporalTransformer(self.dim, self.heads, expand=2)
        # todo:
        # adaptative global average pooling 3d
        self.pool = nn.AdaptiveAvgPool3d((1, 1, dim))
        # self.pool = nn.AdaptiveAvgPool2d((1, dim))
        self.top_dense = nn.Linear(dim, dim * 3)

    def init_weights(self):
        """
        Initiate the parameters either from existing checkpoint or from scratch.
        """
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, inputs):
        """inputs.shape 64,3,32,21,1 (N C T V M)"""
        n, c, t, v, m = inputs.size()
        x = inputs.permute(0, 4, 2, 3, 1).contiguous().view(-1, t, v, c)
        # N, C, T, V = x.size()
        # todo: embedding + n_pos
        x = self.embedding(x)  # 64,32,21,252

        x = self.transformer1(x)  # temporal
        x = self.transformer2(x)  # temporal

        # 64,21,32,252 nvtc ->
        x = self.pool(x)  # 64,1,1,256
        x = x.view(-1, self.dim)  # 64,256
        x = self.top_dense(x)  # 64,756

        return x
