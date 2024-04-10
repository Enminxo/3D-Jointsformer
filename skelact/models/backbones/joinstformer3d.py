import torch
import torch.nn as nn
from mmcv.cnn import normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmaction.models.builder import BACKBONES
from mmaction.utils import get_root_logger

from .resnet3d_slowonly import ResNet3dSlowOnly

# cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
class SkeletonEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # self.patch_size = patch_size
        # self.projection = nn.Conv3d(in_channels=1,
        #                             out_channels=512,  # 256
        #                             kernel_size=(2, 3, 3),
        #                             stride=(2, 3, 3))
        # todo: 3DCNN backbine to embed the input
        self.projection = ResNet3dSlowOnly()
        # todo: output of backbone = 64, 512, 16, 1, 1
        # self.dense = nn.Sequential(
        #     nn.Linear(self.embed_dim * 7, self.embed_dim * 4),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(self.embed_dim * 4, self.embed_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.3)
        # )

    def forward(self, x):
        """
        return projected and flattend patches
        """
        n, c, t, v, m = x.size()
        # x = x.permute(0, 4, 2, 3, 1).contiguous()
        # todo:
        #  filters = embed_dim ; kernel_size = patch_size; strides = patch_size,
        #  out_channels = 512
        y = self.projection(x)  # b,embed_dim,16,1,1
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(n, 16, -1)
        # y = self.projection(x)  # 64,512,16,7,1
        # y = y.view(-1, 16, self.embed_dim*7)
        # y = self.dense(y)

        return y

# def pos_embed(input, d_model):
#     input = input.view(-1, 1).to(cuda_device)  # max_len,1
#     dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)  # 1,256
#     sin = torch.sin(input / 10000 ** (2 * dim / d_model))  # max_len, 256
#     cos = torch.cos(input / 10000 ** (2 * dim / d_model))  # max_len, 256
#
#     out = torch.zeros((input.shape[0], d_model), device=input.device)  # 16,512
#     out[:, ::2] = sin
#     out[:, 1::2] = cos
#     return out
#
# # def sinusoid_encoding_table(max_len, d_model):
# #     pos = torch.arange(max_len, dtype=torch.float32)
# #     out = pos_embed(pos, d_model)
# #     return out
#
# class TemporalEmbedding(nn.Module):
#     def __init__(self, max_len, d_model):
#         super().__init__()
#         self.max_len = max_len
#         self.d_model = d_model
#         # self.joints = joints  # 21
#
#     def forward(self, inputs):
#         pos = torch.arange(self.max_len, dtype=torch.float32)
#         out = pos_embed(pos, self.d_model)
#
#         return inputs + out

class AddPositionEmbedding(nn.Module):
    #  todo: 16 tokens
    def __init__(self, shape=(1, 16, 512)):
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
        self.embed_dim = 2 * dim
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
        # head_dim = 756//6= 126 ; 1024//8 = 128

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

        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(self.dim)
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
class Jointsformer3D(nn.Module):
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
        # todo: skeleton embedding that used 3DCNN as backbone to obtain the projected patches
        self.embedding = SkeletonEmbedding(self.dim)
        # self.embedding = ResNet3dSlowOnly()
        # todo: position embedding
        # self.pos_embedding = TemporalEmbedding(max_len=16, d_model=dim)  # add to the class token
        self.pos_embedding = AddPositionEmbedding() # add to the class token

        self.norm = nn.LayerNorm(self.dim)

        self.transformer1 = TransformerBlock(self.dim, self.heads, expand=4)
        self.transformer2 = TransformerBlock(self.dim, self.heads, expand=4)
        self.transformer3 = TransformerBlock(self.dim, self.heads, expand=4)
        self.transformer4 = TransformerBlock(self.dim, self.heads, expand=4)

        self.pool = nn.AdaptiveAvgPool2d((1, self.dim))
        # self.pool = nn.AdaptiveAvgPool1d(self.dim)
        self.top_dense = nn.Linear(self.dim, self.dim * 2)

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
        # inputs.shape = 64,3,32,21,1
        n, c, t, v, m = inputs.size()
        # todo: projected and flattened patches
        x = self.embedding(inputs)
        # todo: position embedding
        x = self.pos_embedding(x)
        # todo: input to the transformer encoder layers
        x = self.transformer1(x)
        x = self.transformer2(x)
        # x = self.transformer3(x)
        # x = self.transformer4(x)

        # 64,32,dim
        x = self.pool(x).squeeze(dim=1)  # 64,dim
        x = self.top_dense(x)  # 64,dim*2

        return x
