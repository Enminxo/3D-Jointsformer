import torch
import torch.nn as nn

from mmcv.cnn import normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmaction.models.builder import BACKBONES
from mmaction.utils import get_root_logger

# todo: NodeEmbedding applies embedding for each node of the input skeleton

"""
    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 2, 1, 3).contiguous()  # N M T C V
        # keep the rest of the dim the same, operates on each joint separately
        x = x.permute(0, 4, 2, 1, 3).contiguous().view(-1, v)
        
        y = self.dense(x)
"""


class NodeEmbedding(nn.Module):
    def __init__(self, input_units, out_units):
        super().__init__()

        """
        a nivel de nodo 1x3 -> 
        """
        self.input_units = input_units
        self.out_units = out_units  # out_joints = dim
        self.expand_dim = 2 * out_units  # 63*2 = 126
        # Embedding created using a sequential container to concatenate several layers

        self.dense = nn.Sequential(
            nn.Linear(self.input_units, self.expand_dim),  # x2
            nn.GELU(),
            nn.Linear(self.expand_dim, self.out_units))

        # self.dense = nn.Sequential(
        #     nn.Linear(input_shape[-1], self.units),  # the last dim of the input
        #     nn.GELU(),
        #     nn.Linear(self.units, self.units)
        # )

        self.dense.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)

    def forward(self, x):
        # return the embedding of the landmarks
        # cómo hacer embedding para cada nodo
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous()  # N M T V C
        x = x.view(-1, v, c)  # embedding de cada nodo
        y = self.dense(x)
        y = y.view(n, t, v, -1)
        # N T V C
        return y

# todo: add position embedding to each node (n,32,21,xxx) => ns = ns + npos
#



class SkeletonEmbedding(nn.Module):
    def __init__(self, in_joints, out_joints):
        super().__init__()
        self.in_joints = in_joints  # 3*21= 63
        self.out_joints = out_joints  # out_joints = dim
        self.expand_dim = 2 * in_joints  # 63*2 = 126
        # hacerlo mas grande *4, añadir mas capas, dropout, normalization
        self.dense = nn.Sequential(
            nn.Linear(in_joints, self.expand_dim),
            nn.ReLU(),
            nn.Linear(self.expand_dim, self.expand_dim * 2),
            nn.ReLU(),
            nn.Linear(self.expand_dim * 2, out_joints))  # 63 -> 126 -> 252
        # self.dense = nn.Linear(in_joints, out_joints)
        # weight matrix shape(NxM) with orthogonal initialization, ensure that the weights are not correlated
        # self.trans_mat = nn.Parameter(torch.empty(in_joints, out_joints))
        # nn.init.orthogonal_(self.trans_mat)

    def forward(self, x):
        n, c, t, v, m = x.size()
        # x = x.permute(0, 4, 2, 1, 3).contiguous()  # N M T C V
        # x = x.view(-1, t, c, v)  # N*M T C V
        # x = x.view(-1, v)  # keep the rest of the dim the same, operates on each joint separately
        # x = x.view(-1, c * v)  # operates on the whole skeleton
        x = x.permute(0, 4, 2, 1, 3).contiguous().view(-1, c * v)  # (...,63)
        y = self.dense(x)
        # y = torch.matmul(x, self.trans_mat)  # learn whole skeleton ( L x E)
        y = y.view(n, t, -1)  # b,32,embed_dim
        # y = y.view(n, -1, t)  # b,embed_dim,32

        return y


class AddPositionEmbedding(nn.Module):
    def __init__(self, shape=(1, 32, 252)):
        super().__init__()
        self.position = nn.Parameter(torch.rand(shape), requires_grad=True)

    def forward(self, inputs):
        return inputs + torch.tensor(self.position, dtype=inputs.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, h, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()

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
        # 64,32,252
        x = inputs
        b, t = x.shape[:2]
        q = self.q(x)  # 64,32,756
        k = self.k(x)  # 64,32,756
        v = self.v(x)  # 64,32,756
        # head_dim = 756//6= 126

        q = q.view((b, t, self.h, self.embed_dim // self.h)).permute(0, 2, 1, 3)  # b, h, t,head_dim
        k = k.view((b, t, self.h, self.embed_dim // self.h)).permute(0, 2, 1, 3)
        v = v.view((b, t, self.h, self.embed_dim // self.h)).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # b, h, t, t
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop1(attn)

        out = torch.matmul(attn, v)  # b,h,t,dv
        # out = out.permute(0, 2, 1, 3).contiguous().view(b, t, self.h * self.dim)  # to b,t,h,dv -> b,t,h*dv
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, self.embed_dim)  # to b,t,h,dv -> b,t,h*dv
        out = self.proj(out)  # b, t, dim

        return out


# todo: this Temporal Transformer Block aims to perform temporal attention to the input skeletons.
# It's applied for each skeleton separately
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, expand, attn_dropout=0.2, drop_rate=0.2):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.expand = expand

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate)

        self.attn_dropout = attn_dropout
        self.attention = MultiHeadSelfAttention(dim=dim, h=heads, dropout=attn_dropout)

        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        # MLP
        self.expand_dense = nn.Linear(dim, dim * expand, bias=False)  # dim=252 -> out_dim = 252*2
        self.dense = nn.Linear(dim * expand, dim, bias=False)  # 252*2> out_dim = 252

    def forward(self, inputs):
        # 64,32,252 = batch,sequence_length, embedding_size or features
        x = inputs
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
class HandsFormer(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 in_joints,
                 out_joints,
                 pretrained=None):
        super().__init__()

        self.pretrained = pretrained
        self.dim = dim
        self.heads = heads

        self.embedding = SkeletonEmbedding(in_joints, out_joints)
        
        self.pos_embedding = AddPositionEmbedding()  # 64,32,252 the pe creates a tensor of the same shape as the input

        self.norm = nn.LayerNorm(dim)  # operates on the temporal dim

        self.transformer1 = TransformerBlock(self.dim, self.heads, expand=2)
        self.transformer2 = TransformerBlock(self.dim, self.heads, expand=2)
        self.transformer3 = TransformerBlock(self.dim, self.heads, expand=2)
        self.transformer4 = TransformerBlock(self.dim, self.heads, expand=2)

        self.pool = nn.AdaptiveAvgPool2d((1, dim))
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
        # inputs.shape = 64,3,32,21,1 (N C T V M)
        # n, c, t, v, m = inputs.size()
        # skeleton embedding `+ temporal position encoding
        x = self.embedding(inputs)  # 64,32,252 (batch, sequence_len, embedding)
        x = self.pos_embedding(x)

        x = self.transformer1(x)
        x = self.transformer2(x)
        # x = self.transformer3(x)
        # x = self.transformer4(x)

        # 64,32,252
        x = self.pool(x).squeeze(dim=1)  # 64,252
        x = self.top_dense(x)  # 64,dim*3 = 252*3= 756

        return x
