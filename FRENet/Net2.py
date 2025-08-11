import torch.nn as nn
import torch
import torch.nn.functional as F
# from module.dtcn import DeformableTransposedConv2d
# from module.dcn import DeformConv2d
from einops import rearrange
from bb.convnext import convnext_tiny
from bb.mix_transformer import mit_b2
from module.DySample import DySample
# import torch_dct as dct
import math
from torch import Tensor

def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1,d=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s,dilation=d, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            )
def conv1x1_bn_relu(in_planes, out_planes, k=1, s=1, p=0, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            )
class SELayer(nn.Module): # 假设SELayer的定义与之前的回答一致，如果需要请包含之前的SELayer代码
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResSEBlock(nn.Module): # 假设ResSEBlock的定义与之前的回答一致，如果需要请包含之前的ResSEBlock代码
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, reduction)
        self.stride = stride

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class FrequencyEnhancer(nn.Module):
    def __init__(self, in_channels,ratio=8):
        super().__init__()
        # self.ssem = SSEM_Frequency(in_channels)
        self.phase_conv = nn.Sequential(nn.Conv2d(in_channels*2,in_channels//ratio,1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels//ratio,in_channels,1),
                                        nn.Tanh())
        self.resse_sigma = ResSEBlock(in_channels, in_channels)  # 用于计算sigma的ResSE模块
        self.resse_lf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16,1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels,1, bias=False),
            nn.Sigmoid()
        )  # 用于提取低频信息的ResSE模块
        self.out = conv1x1_bn_relu(2*in_channels,in_channels)
        # self.norm = nn.LayerNorm(in_channels)
    def forward(self, x):

        device = x.device

        f0 = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        f = torch.fft.fftshift(f0, dim=(-2, -1))
        amp = torch.abs(f)
        phase = torch.angle(f)
        B,C,H,W = amp.shape
        resse_m = self.resse_sigma(amp)
        l = min(H, W)
        sigma = torch.minimum(torch.abs(resse_m) + l / 2, torch.tensor(l).float().to(resse_m.device))  # 确保sigma不小于0
        u = torch.arange(W, device=x.device) - W // 2
        v = torch.arange(H, device=x.device) - H // 2
        uu, vv = torch.meshgrid(v, u, indexing='ij')
        d_squared = uu ** 2 + vv ** 2

        hpf = 1 - torch.exp(-d_squared.float() / (2 * sigma ** 2 + 1e-6))
        m_prime = hpf * amp

        phase_cat = torch.concat((torch.sin(phase),torch.cos(phase)),dim=1)
        phase1 = self.phase_conv(phase_cat) * torch.pi

        en = torch.polar(m_prime.float(),phase1.float())

        x_hf = torch.fft.ifftshift(en, dim=(-2, -1)) # 逆向移回零频分量
        x_hf = torch.fft.ifft2(x_hf, dim=(-2, -1), norm='ortho').real
        x_out = torch.concat((self.resse_lf(m_prime)*x,x_hf),dim=1)
        return self.out(x_out)






import torch.fft



class FM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = conv1x1_bn_relu(2*channels,channels)
        self.Fe = FrequencyEnhancer(channels)
        # self.Ee = EdgeConvSep(channels,channels)
    def forward(self,fu):
        fu = self.down(fu)
        fu = self.Fe(fu.to(torch.float32))
        # return self.Ee(fu)+fu
        return fu


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

# class BottConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
#         super(BottConv, self).__init__()
#         self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
#         self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
#         self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
#
#     def forward(self, x):
#         x = self.pointwise_1(x)
#         x = self.depthwise(x)
#         x = self.pointwise_2(x)
#         return x
#
#
# def get_norm_layer(norm_type, channels, num_groups):
#     if norm_type == 'GN':
#         return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
#     else:
#         return nn.InstanceNorm3d(channels)
#
#
# class GBC(nn.Module):
#     def __init__(self, in_channels, norm_type='GN'):
#         super(GBC, self).__init__()
#
#         self.block1 = nn.Sequential(
#             BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
#             get_norm_layer(norm_type, in_channels, in_channels // 16),
#             nn.ReLU()
#         )
#
#         self.block2 = nn.Sequential(
#             BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
#             get_norm_layer(norm_type, in_channels, in_channels // 16),
#             nn.ReLU()
#         )
#
#         self.block3 = nn.Sequential(
#             BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
#             get_norm_layer(norm_type, in_channels, in_channels // 16),
#             nn.ReLU()
#         )
#
#         self.block4 = nn.Sequential(
#             BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
#             get_norm_layer(norm_type, in_channels, 16),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         residual = x
#
#         x1 = self.block1(x)
#         x1 = self.block2(x1)
#         x2 = self.block3(x)
#         x = x1 * x2
#         x = self.block4(x)
#
#         return x + residual

class objenh(nn.Module):
    def __init__(self, in_channels, out_channels,scale):
        super(objenh, self).__init__()
        self.downc = conv1x1_bn_relu(in_channels,out_channels)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.lateral = conv1x1_bn_relu(384, out_channels)
    def forward(self, x,x4):
        x = self.downc(x)

        return x-self.lateral(self.up(x4))

class objenh1(nn.Module):
    def __init__(self, in_channels, out_channels,scale):
        super(objenh1, self).__init__()
        self.downc = conv1x1_bn_relu(in_channels,out_channels)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.lateral = conv1x1_bn_relu(384, out_channels)
    def forward(self, x,x4):
        x = self.downc(x)

        return x+self.lateral(self.up(x4))

# class SegFormerHead(nn.Module):  # 定义SegFormer的头部模块类，继承自nn.Module
#     def __init__(self, num_classes=6, in_channels=[96, 96, 192, 384], embedding_dim=256, dropout_ratio=0.1):
#         super(SegFormerHead, self).__init__()  # 调用父类nn.Module的初始化函数
#         # 对每一层的输入通道数进行解构
#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
#
#         # 为每一层定义一个MLP模块，用于学习抽象表示
#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
#         self.gbc = GBC(embedding_dim)
#         # 定义一个卷积模块，用于融合四层的特征表示
#         self.linear_fuse = conv1x1_bn_relu(embedding_dim * 4, embedding_dim)
#         # 定义一个卷积层，用于最终的预测
#         self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
#         self.bound_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)
#         # 定义一个dropout层，用于防止过拟合
#         self.dropout = nn.Dropout2d(dropout_ratio)
#
#     def forward(self, inputs):  # 定义SegFormer头部模块类的前向传播函数
#         c1, c2, c3, c4 = inputs  # 对输入特征进行解构
#
#         # 对每一层的特征进行解码
#
#         n, _, h, w = c4.shape  # 从c4的形状中获取batch大小n，高度h和宽度w
#         # 对c4特征进行MLP处理，并改变维度顺序
#         _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         # 将_c4的大小上采样到c1的大小
#         _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
#
#         # 对c3特征进行MLP处理，并改变维度顺序
#         _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         # 将_c3的大小上采样到c1的大小
#         _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
#
#         # 对c2特征进行MLP处理，并改变维度顺序
#         _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         # 将_c2的大小上采样到c1的大小
#         _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
#
#         # 对c1特征进行MLP处理，并改变维度顺序
#         _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
#         # 将四层的特征进行拼接，然后通过卷积模块进行融合
#         _c = self.gbc(self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)))
#
#         x = self.dropout(_c)  # 对融合后的特征进行dropout操作
#         out = self.linear_pred(x)  # 对dropout后的特征进行最终预测
#         bound = self.bound_pred(x)  # 对dropout后的特征进行最终预测
#
#         return out,bound  # 返回预测结果
# class PPM(nn.ModuleList):
#     def __init__(self, pool_sizes, in_channels, out_channels):
#         super(PPM, self).__init__()
#         self.pool_sizes = pool_sizes
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         for pool_size in pool_sizes:
#             self.append(
#                 nn.Sequential(
#                     nn.AdaptiveMaxPool2d(pool_size),
#                     nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
#                 )
#             )
#
#     def forward(self, x):
#         out_puts = []
#         for ppm in self:
#             ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
#                                                 align_corners=True)
#             out_puts.append(ppm_out)
#         return out_puts

class PATM_BAB(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, dilation_1=3, dilation_2=2):
        super().__init__()
        self.conv1 = conv3x3_bn_relu(channel_1, channel_2)
        # self.conv1_Dila = conv3x3_bn_relu(channel_2, channel_2)
        self.conv_fuse = conv1x1_bn_relu(channel_2 *2, channel_2)
        # self.drop = nn.Dropout(0.3)
    def forward(self, x):
        x1 = self.conv1(x)
        # x1_dila = self.conv1_Dila(x1)
        x1 = torch.cat([x1 * torch.cos(x1), x1 * torch.sin(x1)], dim=1)
        # print('x1_dila + x2_dila+x3',x1_dila.shape)
        x_fuse = self.conv_fuse(x1)
        # x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))
        # print('x_f',x_fuse.shape)
        # x_fuse= self.drop(x_fuse)
        # print()
        return x_fuse

class dfpnHead(nn.Module):  # 定义SegFormer的头部模块类，继承自nn.Module
    def __init__(self, num_classes=6, in_channels=[96, 96, 192],out_channel=192):
        super(dfpnHead, self).__init__()  # 调用父类nn.Module的初始化函数
        # 对每一层的输入通道数进行解构
        c1_in_channels, c2_in_channels, c3_in_channels = in_channels
        # self.d4 = PATM_BAB(c4_in_channels,c4_in_channels,c3_in_channels)
        self.d3 = PATM_BAB(out_channel,out_channel)
        self.d2 = PATM_BAB(out_channel,out_channel)
        self.d1 = PATM_BAB(out_channel,out_channel)
        self.lateral3 = conv1x1_bn_relu(c3_in_channels,out_channel)
        self.lateral2 = conv1x1_bn_relu(c2_in_channels,out_channel)
        self.lateral1 = conv1x1_bn_relu(c1_in_channels,out_channel)
        self.out = nn.Conv2d(out_channel,num_classes,1)
        self.out1 = nn.Conv2d(out_channel,num_classes,1)
        self.out2 = nn.Conv2d(out_channel,num_classes,1)
        self.outb = nn.Conv2d(out_channel,2,1)
        self.fuseconv = conv1x1_bn_relu(3*out_channel,out_channel)

        self.up = nn.UpsamplingBilinear2d(size=(480,640))
        self.up2 = DySample(out_channel,scale=2)
        self.up21 = DySample(out_channel,scale=2)
        self.up22 = DySample(out_channel,scale=2)
        self.up4 = DySample(out_channel,scale=4)
    def forward(self, inputs):  # 定义SegFormer头部模块类的前向传播函数
        c1, c2, c3 = inputs  # 对输入特征进行解构
        d3 = self.d3(self.lateral3(c3))
        d2 = self.d2(self.up2(d3)+self.lateral2(c2))
        d1 = self.d1(self.up21(d2)+self.lateral1(c1))
        out = self.out(d1)
        bound = self.outb(d1)
        aux1 = self.out1(d2)
        aux2 = self.out2(d3)


        return self.up(out),self.up(aux1),self.up(aux2),self.up(bound)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # self.cbackbone = mit_b2()#64,128,320,512
        # self.cbackbone.load_state_dict(torch.load("/media/xug/shuju/SUS/bb/pth/mit_b2.pth"),strict=False)
        # self.cbackbone1 = mit_b2()
        # self.cbackbone1.load_state_dict(torch.load("/media/xug/shuju/SUS/bb/pth/mit_b2.pth"), strict=False)
        self.cbackbone = convnext_tiny(True)
        self.cbackbone1 = convnext_tiny(True)#96,192,384,768
        self.l4 = conv1x1_bn_relu(768,384)
        self.l41 = conv1x1_bn_relu(768,384)

        self.obje1 = objenh(96,96,8)
        self.obje2 = objenh(192,96,4)
        self.obje3 = objenh(384,192,2)
        self.obje11 = objenh1(96,96,8)
        self.obje21 = objenh1(192,96,4)
        self.obje31 = objenh1(384,192,2)
        # self.fu4 = DCTFusion(512,320)
        self.fm3 = FM(192)
        self.fm2 = FM(96)
        self.fm1 = FM(96)
        self.up = nn.UpsamplingBilinear2d(size=(480, 640))
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.down2 = conv3x3_bn_relu(96, 96, s=2, p=1)
        self.dillconv = conv1x1_bn_relu(96+96+192,128)
        # self.de = decoder()
        self.de = dfpnHead()
    def forward(self, r,t):

        r1, r2, r3, r4 = self.cbackbone(r)
        r4 = self.l4(r4)

        t1, t2, t3, t4 = self.cbackbone1(t)
        t4 = self.l41(t4)
        fu1 = self.fm1(torch.concat((self.obje1(r1,r4),self.obje11(t1,t4)),dim=1))
        fu2 = self.fm2(torch.concat((self.obje2(r2,r4),self.obje21(t2,t4)),dim=1))
        fu3 = self.fm3(torch.concat((self.obje3(r3,r4),self.obje31(t3,t4)),dim=1))
        # fu4 = (r4+t4)


        # out1,out2 = self.de(fu1,fu2,fu3,fu4)
        out,aux1,aux2,b = self.de((fu1,fu2,fu3))
        dill = self.dillconv(torch.concat((self.down2(fu1), fu2, self.up2(fu3)), dim=1))
        # return  self.up(out1),self.up(out2),self.up(out3)
        return  out,aux1,aux2,b,dill


if __name__ == "__main__":
    from thop import profile
    a = torch.randn(1, 3, 480, 640).cuda()
    b = torch.randn(1, 3, 480, 640).cuda()

    model = Net2().cuda()
    flops,params = profile(model,(a,b,))
    print('flops: %.2f M,params: %.2f M'%(flops/1000000.0,params/1000000.0))
    out = model(a,b)

    for i in range(len(out)):
        print(out[i].shape)
     