import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
def conv1x1_bn_relu(in_planes, out_planes, k=1, s=1, p=0, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            )
def cosin_sim(f1,f2):
    f1_flat = f1.view(f1.size(0),f1.size(1),-1)
    f2_flat = f2.view(f2.size(0),f2.size(1),-1)
    return F.cosine_similarity(f1_flat,f2_flat,dim=1)

def single_sample_cka(feat_s, feat_t):
    """输入为单个样本的特征，shape [C, H, W] 或 [C]"""
    x = feat_s.flatten(1) if feat_s.dim() > 1 else feat_s.unsqueeze(0)
    y = feat_t.flatten(1) if feat_t.dim() > 1 else feat_t.unsqueeze(0)
    # Center Gram
    def center_gram(gram):
        n = gram.size(0)
        unit = torch.ones(n, n, device=gram.device) / n
        return gram - unit @ gram - gram @ unit + unit @ gram @ unit
    def gram_linear(x):
        return x @ x.t()
    gx = center_gram(gram_linear(x))
    gy = center_gram(gram_linear(y))
    cka = (gx * gy).sum() / (
        torch.norm(gx, p='fro') * torch.norm(gy, p='fro') + 1e-8
    )
    return cka.item()

# 支持batch评估: 假如每个样本的特征为[B, C, H, W]
def batch_cka_scores(feat_s, feat_t):
    """
    对每个样本分别计算CKA作为难度分数
    返回: [B]个样本的分数(1-cka越大，难度越高)
    """
    B = feat_s.shape[0]
    difficulites = []
    for i in range(B):
        cka = single_sample_cka(feat_s[i], feat_t[i])
        difficulites.append(1 - cka)
    return torch.tensor(difficulites)

def cka(features_x, features_y):
    """
    计算两个特征之间的Centered Kernel Alignment (CKA)相似度

    参数:
        features_x (torch.Tensor): 形状为(batch_size, channels, height, width)
        features_y (torch.Tensor): 形状与features_x相同

    返回:
        cka_value (float): 范围在[0, 1]之间的相似度
    """
    # 将特征转换为二维矩阵 (num_samples, channels)
    X = features_x.view(features_x.size(0), features_x.size(1), -1).permute(0, 2, 1).contiguous().view(-1,
                                                                                                       features_x.size(
                                                                                                           1))
    Y = features_y.view(features_y.size(0), features_y.size(1), -1).permute(0, 2, 1).contiguous().view(-1,
                                                                                                       features_y.size(
                                                                                                           1))

    # 中心化特征矩阵
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # 计算协方差矩阵
    cov_xy = X_centered.T @ Y_centered
    cov_xx = X_centered.T @ X_centered
    cov_yy = Y_centered.T @ Y_centered

    # 计算HSIC
    n = X.size(0)
    hsic_xy = (cov_xy ** 2).sum() / (n - 1) ** 2
    hsic_xx = (cov_xx ** 2).sum() / (n - 1) ** 2
    hsic_yy = (cov_yy ** 2).sum() / (n - 1) ** 2

    # 计算CKA并确保数值稳定性
    cka_value = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)
    return torch.clamp(cka_value, 0.0, 1.0).item()
class freAttention(nn.Module):

    def __init__(self, dim, qkv_bias=False, qk_scale=None):
        super().__init__()
        # self.p = nn.Sequential(nn.Conv2d(dim1, dim2, 1), nn.ReLU(inplace=True))
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=qkv_bias)
        self.proj = conv1x1_bn_relu(dim,dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.attn_drop = nn.Dropout(0.1)

    def forward(self, x1, x2):
        # x1 = self.p(x1)  # (B, dim2, H, W)
        B, C, H, W = x1.shape

        # LayerNorm after flatten
        x1_flat = rearrange(x1, 'b c h w -> b (h w) c')
        x1_norm = self.norm1(x1_flat)
        x1 = rearrange(x1_norm, 'b (h w) c -> b c h w', h=H, w=W)

        x2_flat = rearrange(x2, 'b c h w -> b (h w) c')
        x2_norm = self.norm2(x2_flat)
        x2 = rearrange(x2_norm, 'b (h w) c -> b c h w', h=H, w=W)

        q = self.q(x1)
        kv = self.kv(x2)
        k, v = kv.chunk(2, dim=1)

        # 频域操作
        q_freq = torch.fft.fft2(q.float(), dim=(-2, -1), norm='ortho')
        k_freq = torch.fft.fft2(k.float(), dim=(-2, -1), norm='ortho')
        attn_freq = q_freq * k_freq
        attn = torch.fft.ifft2(attn_freq, dim=(-2, -1), norm='ortho').real

        attn = rearrange(attn, 'b c h w -> b (h w) c')
        attn = F.layer_norm(attn, [self.dim], weight=self.weight, bias=self.bias)
        attn = rearrange(attn, 'b (h w) c -> b c h w', h=H, w=W)
        # attn = self.attn_drop(attn)

        x = attn * v
        out = self.proj(x)
        return out

class spaAttention(nn.Module):

    def __init__(self, dim,num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        # self.p = nn.Sequential(nn.Conv2d(dim,dim,1),nn.ReLU(inplace=True))
        self.dim = dim

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.sr_ratio = i
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim,dim)
        # if i > 1:
        self.sr = nn.Conv2d(dim, dim, kernel_size=10, stride=10)
        self.norm = nn.LayerNorm(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x1,x2):
        # x1 = self.p(x1)
        # print(x1.shape)

        B, C, H, W = x1.shape
        x1_flat = x1.reshape(B, C, -1).permute(0, 2, 1)
        x1_flat = self.norm1(x1_flat)
        q = self.q(x1_flat).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # if self.sr_ratio > 1:
        x_ = self.sr(x2).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # else:
        # x2_flat = self.norm2(x2.reshape(B, C, -1).permute(0, 2, 1))
        # kv = self.kv(x2_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # k,v = self.kv(x2_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # print(q.shape)
        # print(k.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C).contiguous()
        # print(x.shape)
        return self.proj(x).permute(0,2,1).reshape(B,C,H,W).contiguous()
#









class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x






class FASML1(nn.Module):
    """Feature-Adaptive Self-Paced Mutual Learning for Heterogeneous Networks"""

    def __init__(self, channels,temperature=0.5):
        super(FASML1, self).__init__()

        self.temperature = temperature
        self.current_epoch = 0
        self.max_epochs = 150  # 可配置
        # --- EMA 相关初始化 ---
        self.num_levels = 1  # 确保 num_levels 被设置
        self.register_buffer('difficulty_ema', torch.zeros(self.num_levels))
        self.ema_beta = 0.98  # EMA 更新的平滑因子 (beta 越大，历史权重越高)
        # 用于跟踪每个层级的 EMA 是否已经用第一个批次的均值初始化
        self.ema_initialized = [False] * self.num_levels

        # 特征适配层 (CNN -> Transformer)
        self.cnn2trans = freAttention(channels)



    def set_epoch(self, epoch, max_epochs=None):
        self.current_epoch = epoch
        if max_epochs is not None:
            self.max_epochs = max_epochs

    def calculate_feature_difficulty2(self, f1, f2):
        """计算特征的难度分数"""

        similarity_score = batch_cka_scores(f1,f2).cuda()

        # f1_var = f1.var(dim=[2, 3])
        # f2_var = f2.var(dim=[2, 3])
        # complexity_score = (f1_var + f2_var) / 2
        # print(similarity_score)
        # 综合难度分数
        difficulty = similarity_score

        return difficulty

    def update_difficulty_ema(self, current_difficulty_mean):
        """更新指定层级的难度 EMA 值"""
        # 如果是该层级第一次更新，直接用当前均值初始化 EMA
        if not self.ema_initialized:
             self.difficulty_ema = current_difficulty_mean
             self.ema_initialized = True
        else:
            # EMA 更新公式: new_ema = beta * old_ema + (1 - beta) * current_value
            self.difficulty_ema = self.ema_beta * self.difficulty_ema + \
                                             (1 - self.ema_beta) * current_difficulty_mean

    def get_spl_weights(self, difficulty):
        """根据 EMA 难度阈值计算 SPL 权重"""
        # 如果当前批次不为空，用当前批次的平均难度更新 EMA
        if difficulty.numel() > 0:
            # detach() 防止梯度流经 EMA 更新路径
            current_mean_difficulty = difficulty.detach().mean()
            # 更新对应层级的 EMA
            self.update_difficulty_ema(current_mean_difficulty.float())

        threshold = (1+0.001)**self.current_epoch
        # 备选方案：基于 EMA 附近假定分布的固定分位数？更复杂。
        difficulty = difficulty/self.difficulty_ema
        # print(difficulty)
        # 生成权重：简单样本 (难度 <= 阈值) 获得权重 1.0
        weights = torch.ones_like(difficulty)
        # 找到困难样本的索引 (难度 > 阈值)
        hard_indices = difficulty > threshold
        easy_indices = difficulty<=threshold
        # 困难样本的权重随训练进度增加
        # 权重从 0.05 线性增加到 1.0
        weights[hard_indices] = 0
        weights[easy_indices] = 1
        # 将权重调整为可广播的形状：(batch_size, 1, 1, 1)
        return weights.view(-1, 1, 1, 1)

    def forward(self, cf, tf):
        """执行特征自适应自步相互学习"""
        # cftotal_loss = 0
        tftotal_loss = 0
        feature_weights = []
        # print(self.current_epoch)
        # for i, (cf, tf) in enumerate(zip(cnn_features, transformer_features)):

        cf2t = self.cnn2trans(cf,tf)

        tf_no = F.normalize(tf,p=2,dim=1)
        # 预测特征是经过适配的 Transformer 特征
        cf2t_no = F.normalize(cf2t,p=2,dim=1)

        tf_difficulty = self.calculate_feature_difficulty2(tf_no.detach(), cf2t_no.detach())

        tf_weights = self.get_spl_weights(tf_difficulty)






        loss_per_pixel = F.mse_loss(tf_no, cf2t_no, reduction='none')

        weighted_loss_per_pixel = loss_per_pixel * tf_weights

        tf_loss = weighted_loss_per_pixel.mean()

        tftotal_loss += (tf_loss)

        return tftotal_loss

class FASML2(nn.Module):
    """Feature-Adaptive Self-Paced Mutual Learning for Heterogeneous Networks"""

    def __init__(self,  channels, temperature=0.5):
        super(FASML2, self).__init__()

        self.temperature = temperature
        self.current_epoch = 0
        self.max_epochs = 150  # 可配置
        # --- EMA 相关初始化 ---
        self.num_levels = 1  # 确保 num_levels 被设置
        self.register_buffer('difficulty_ema', torch.zeros(self.num_levels))
        self.ema_beta = 0.98  # EMA 更新的平滑因子 (beta 越大，历史权重越高)
        self.ema_initialized = [False] * self.num_levels

        # # 特征适配层 (Transformer -> CNN)
        self.trans2cnn = spaAttention(channels)




    def set_epoch(self, epoch, max_epochs=None):
        self.current_epoch = epoch
        if max_epochs is not None:
            self.max_epochs = max_epochs

    def calculate_feature_difficulty1(self, f1, f2):
        """计算特征的难度分数"""
        # 通过特征相似度预测器评估特征相似度
        # f1_resized = F.interpolate(f1, size=f2.shape[2:], mode='bilinear', align_corners=False)

        # # 特征拼接并通过预测器
        # concat_feat = torch.cat([f1_resized, f2], dim=1)
        # similarity_score = self.similarity_predictor2[level_idx](concat_feat)
        # print(f1.shape)

        similarity_score = batch_cka_scores(f1,f2).cuda()
        # 计算特征统计信息作为难度指标
        # f1_var = f1.var(dim=[2, 3])
        # f2_var = f2.var(dim=[2, 3])
        # complexity_score = (f1_var + f2_var) / 2

        # 综合难度分数
        difficulty = similarity_score

        return difficulty

    def update_difficulty_ema(self, current_difficulty_mean):
        """更新指定层级的难度 EMA 值"""
        # 如果是该层级第一次更新，直接用当前均值初始化 EMA
        # print(self.ema_initialized)
        if not self.ema_initialized:
             self.difficulty_ema = current_difficulty_mean
             self.ema_initialized = True
        else:
            # EMA 更新公式: new_ema = beta * old_ema + (1 - beta) * current_value
            # print(self.difficulty_ema[level_idx])
            # print(current_difficulty_mean)
            self.difficulty_ema = self.ema_beta * self.difficulty_ema + \
                                             (1 - self.ema_beta) * current_difficulty_mean

    def get_spl_weights(self, difficulty):
        """根据 EMA 难度阈值计算 SPL 权重"""
        # 如果当前批次不为空，用当前批次的平均难度更新 EMA
        if difficulty.numel() > 0:
            # detach() 防止梯度流经 EMA 更新路径
            current_mean_difficulty = difficulty.detach().mean()
            # 更新对应层级的 EMA
            self.update_difficulty_ema(current_mean_difficulty.float())

        threshold = (1+0.001)**self.current_epoch
        # 备选方案：基于 EMA 附近假定分布的固定分位数？更复杂。
        difficulty = difficulty/self.difficulty_ema
        # 生成权重：简单样本 (难度 <= 阈值) 获得权重 1.0
        weights = torch.ones_like(difficulty)
        # 找到困难样本的索引 (难度 > 阈值)
        hard_indices = difficulty > threshold
        easy_indices = difficulty<=threshold
        # 困难样本的权重随训练进度增加
        # 权重从 0.05 线性增加到 1.0
        weights[hard_indices] = 0
        weights[easy_indices] = 1

        return weights.view(-1, 1, 1, 1)

    def forward(self, transformer_features, cnn_features):
        """执行前向传播，包含特征适配和基于 EMA 的 SPL 加权"""
        cftotal_loss = 0.0 # 初始化总损失
        # 确保输入的特征列表层级数量一致


        tf2c = self.trans2cnn(transformer_features, cnn_features)

            # 目标特征是原始 CNN 特征
        cf_no = F.normalize(cnn_features,p=2,dim=1)
            # 预测特征是经过适配的 Transformer 特征
        tf2c_no = F.normalize(tf2c,p=2,dim=1)
            # print(i)
        cf_difficulty = self.calculate_feature_difficulty1(cf_no.detach(), tf2c_no.detach()) # 调用（可能在父类中定义的）难度计算
            # print(cf_difficulty)
        cf_weights = self.get_spl_weights(cf_difficulty) # 调用本类重写的 get_spl_weights
        loss_per_pixel = F.mse_loss(tf2c_no, cf_no, reduction='none')
        weighted_loss_per_pixel = loss_per_pixel * cf_weights
        cf_loss = weighted_loss_per_pixel.mean()

        cftotal_loss += (cf_loss)

        return cftotal_loss # 返回最终的总损失