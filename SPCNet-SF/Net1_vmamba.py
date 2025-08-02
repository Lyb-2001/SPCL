import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from bb.mix_transformer import mit_b2
from bb.convnext import convnext_tiny
from model_others.Net1.vmamba import ConcatMambaFusionBlock,Channel_SS2D,VSSBlock,ConMB_SS2D,Backbone_VSSM
# from model_others.Net1.transformer import Attention
from model_others.Net1.decoder import MaskTransformer
from model_others.Net1.DySample import DySample
from torch import einsum
from einops import rearrange
from functools import partial
from typing import Optional, Callable, Any


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1,g=1,d=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p,groups=g, bias=b,dilation=d),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            )
def conv1x1_bn_relu(in_planes, out_planes, k=1, s=1, p=0, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            )


class fusion(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(fusion, self).__init__()
        self.convd1 = conv1x1_bn_relu(in_channel,out_channel)
        self.convd2 = conv1x1_bn_relu(in_channel,out_channel)
        # self.conv = conv1x1_bn_relu(out_channel,out_channel)
        self.sim = ConcatMambaFusionBlock(out_channel)


    def forward(self, rgb,t):
        b,c,h,w = rgb.size()
        rgb = self.convd1(rgb)
        t = self.convd2(t)

        out2 = self.sim(rgb.permute(0, 2, 3, 1),t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out2


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Rdeu(nn.Module):
    def __init__(self, in_channels):
        super(Rdeu, self).__init__()
        self.block1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,groups=in_channels,dilation=1)
        self.block2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=2,groups=in_channels,dilation=2)
        self.block3 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=3,groups=in_channels,dilation=3)
        # self.block1 = conv3x3_bn_relu(in_channels,in_channels,k=3,s=1,p=1,g=in_channels,d=1)
        # self.block2 = conv3x3_bn_relu(in_channels,in_channels,k=3,s=1,p=2,g=in_channels,d=2)
        # self.block3 = conv3x3_bn_relu(in_channels,in_channels,k=3,s=1,p=3,g=in_channels,d=3)
        self.fuse = nn.Sequential(nn.BatchNorm2d(in_channels),nn.ReLU(),conv1x1_bn_relu(in_channels,in_channels),
                                  nn.Conv2d(in_channels,in_channels,1),nn.BatchNorm2d(in_channels))

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        out = x1+x2+x3
        return F.relu(self.fuse(out)+x)

class SegFormerHead(nn.Module):  # 定义SegFormer的头部模块类，继承自nn.Module
    def __init__(self, num_classes=6, in_channels=[96, 96, 192, 384], embedding_dim=384, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()  # 调用父类nn.Module的初始化函数
        # 对每一层的输入通道数进行解构
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # 为每一层定义一个MLP模块，用于学习抽象表示
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        # self.linear_out = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        # self.gbc = GBC(embedding_dim)
        # 定义一个卷积模块，用于融合四层的特征表示
        self.linear_fuse = conv1x1_bn_relu(embedding_dim * 4, embedding_dim)
        # 定义一个卷积层，用于最终的预测
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.bird_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)
        # self.bound_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)
        # 定义一个dropout层，用于防止过拟合
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.up2 = DySample(in_channels=embedding_dim,scale=2)
        self.up4 = DySample(in_channels=embedding_dim,scale=4)
        self.up8 = DySample(in_channels=embedding_dim,scale=8)
        self.channelssm = Channel_SS2D(4*embedding_dim,dmodel=64)
    def forward(self, inputs):  # 定义SegFormer头部模块类的前向传播函数
        c1, c2, c3, c4 = inputs  # 对输入特征进行解构
        # 对每一层的特征进行解码

        n, _, h, w = c4.shape  # 从c4的形状中获取batch大小n，高度h和宽度w
        # 对c4特征进行MLP处理，并改变维度顺序
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.up8(_c4)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.up4(_c3)
        # 对c2特征进行MLP处理，并改变维度顺序
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c2 = self.up2(_c2)
        # 对c1特征进行MLP处理，并改变维度顺序
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # 将四层的特征进行拼接，然后通过卷积模块进行融合
        _c = self.channelssm(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.linear_fuse(_c)

        x = self.dropout(_c)# 对融合后的特征进行dropout操作

        out = self.linear_pred(x)  # 对dropout后的特征进行最终预测
        aux = self.bird_pred(x)
        return out,aux  # 返回预测结果

class TR(nn.Module):  # 定义SegFormer的头部模块类，继承自nn.Module
    def __init__(self, num_classes=1, in_channels=[96, 96, 192, 384], embedding_dim=64):
        super(TR, self).__init__()  # 调用父类nn.Module的初始化函数
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.c1 = conv1x1_bn_relu(c1_in_channels,embedding_dim)
        self.c2 = conv1x1_bn_relu(c2_in_channels,embedding_dim)
        self.c3 = conv1x1_bn_relu(c3_in_channels,embedding_dim)
        self.c4 = conv1x1_bn_relu(c4_in_channels,embedding_dim)
        self.c11 = Rdeu(embedding_dim)
        self.c22 = Rdeu(embedding_dim)
        self.c33 = Rdeu(embedding_dim)
        # self.c11 = Rdeu(embedding_dim)
        # self.c22 = Rdeu(embedding_dim)
        # self.c33 = Rdeu(embedding_dim)
        self.out = nn.Sequential(conv3x3_bn_relu(embedding_dim,embedding_dim),nn.Conv2d(embedding_dim,1,1))
        self.up1 = DySample(embedding_dim,scale=2)
        self.up2 = DySample(embedding_dim,scale=2)
        self.up3 = DySample(embedding_dim,scale=2)
        # self.ca = SELayer(embedding_dim)
    def forward(self, inputs):  # 定义SegFormer头部模块类的前向传播函数
        c1, c2, c3, c4 = inputs  # 对输入特征进行解构
        c1 = self.c1(c1)
        c2 = self.c2(c2)
        c3 = self.c3(c3)
        c4 = self.c4(c4)
        # c4 = self.ca(c4)+c4
        de3 = self.c33(self.up1(c4)+c3)
        de2 = self.c22(self.up2(de3)+c2)
        de1 = self.c11(self.up3(de2)+c1)

        return self.out(de1)  # 返回预测结果



class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.backbone1 = Backbone_VSSM(
            pretrained='/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/1SUS/bb/pth/vssmtiny_dp01_ckpt_epoch_292.pth',
            norm_layer=nn.LayerNorm,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=96,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )#64,192,320,512
        # self.backbone1.load_state_dict(torch.load("/home/data/SUS/bb/pr/mit_b2.pth"),strict=False)
        self.backbone2 = Backbone_VSSM(
            pretrained='/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/1SUS/bb/pth/vssmtiny_dp01_ckpt_epoch_292.pth',
            norm_layer=nn.LayerNorm,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=96,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )
        # self.backbone2.load_state_dict(torch.load("/home/data/SUS/bb/pr/mit_b2.pth"), strict=False)
        # self.backbone1 = convnext_tiny(True)#96,192,384,768
        # self.backbone2 = convnext_tiny(True)
        self.up = nn.UpsamplingBilinear2d(size=(480,640))
        # self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        # self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fu4 = fusion(768,384)
        self.d1 = conv1x1_bn_relu(96,96)
        self.d2 = conv1x1_bn_relu(192,96)
        self.d3 = conv1x1_bn_relu(384,192)
        self.itr = TR()
        self.de = SegFormerHead()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.down2 = conv3x3_bn_relu(96,96,s=2,p=1)
        self.dillconv = conv1x1_bn_relu(96+96+192,128)
    def forward(self, r,t):
        r1, r2, r3, r4 = self.backbone1(r)
        t1, t2, t3, t4 = self.backbone2(t)
        r1 = self.d1(r1)
        r2 = self.d2(r2)
        r3 = self.d3(r3)
        fu4 = self.fu4(r4,t4)
        TR = self.itr((r1,r2,r3,fu4))
        out,aux = self.de((r1,r2,r3,fu4))
        # dill = self.dillconv(torch.concat((self.down2(r1), r2, self.up2(r3)), dim=1))


        return self.up(out),self.up(aux),self.up(TR)#,dill


# if __name__ == "__main__":
#     from thop import profile
#     a = torch.randn(1, 3, 480, 640).cuda()
#     b = torch.randn(1, 3, 480, 640).cuda()
#
#     model = Net1().cuda()
#     # flops,params = profile(model,(a,b,))
#     # print('flops: %.2f M,params: %.2f M'%(flops/1000000.0,params/1000000.0))
#     out = model(a,b)
#     from ptflops import get_model_complexity_info
#     flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
#     print('Flops ' + flops)
#     print('Params ' + params)
#     print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
#     for i in range(len(out)):
#         print(out[i].shape)
if __name__ == "__main__":
    from thop import profile
    a = torch.randn(1, 3, 480, 640).cuda()
    b = torch.randn(1, 3, 480, 640).cuda()

    model = Net1().cuda()
    flops,params = profile(model,(a,b,))
    print('flops: %.2f M,params: %.2f M'%(flops/1000000.0,params/1000000.0))
    out = model(a,b)

    for i in range(len(out)):
        print(out[i].shape)