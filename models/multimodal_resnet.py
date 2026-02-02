import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import MultiheadAttention


class SincConv_fast_my(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=0.5, min_band_hz=50):
        super().__init__()
        if in_channels != 1:
            raise ValueError(f'SincConv only support one input channel (here, in_channels = {in_channels})')
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.stride = stride
        if padding == 'same':
            self.padding = (self.kernel_size - 1) // 2
        else:
            self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        low_hz = min_low_hz
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)
        
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2*math.pi*n_lin/self.kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_/2)) * self.window_
        band_pass_center = 2*band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2*band[:, None])
        
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return nn.functional.conv1d(waveforms, self.filters, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation,
                                   bias=None, groups=1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SingleResNetBranch(nn.Module):
    def __init__(self, cfg, block, layers, token_len, token_dim):
        super(SingleResNetBranch, self).__init__()
        self.signal_channels = cfg['channel_num']
        self.cnn_N_filt = cfg['cnn_N_filt']
        self.cnn_len_filt = cfg['cnn_len_filt']
        self.fs = cfg['fs']
        self.input_dim = cfg['input_dim']
        
        self.layer_norm = nn.LayerNorm(normalized_shape=(1, self.input_dim))
        self.sinc = SincConv_fast_my(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs, 
                                     padding='same', min_low_hz=0.5, min_band_hz=4)
        
        self.conv1 = nn.Conv1d(self.cnn_N_filt[0] * self.signal_channels, 64, 
                              kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.in_channels = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.token_len = token_len
        self.token_dim = token_dim
        self.adapool = nn.AdaptiveAvgPool1d(self.token_len)
        self.token_proj = nn.Linear(512, self.token_dim)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers_list = []
        layers_list.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers_list.append(block(out_channels, out_channels))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        batch = x.size(0)
        x = x.reshape(batch * self.signal_channels, 1, -1)
        x = self.layer_norm(x)
        x = self.sinc(x)
        x = x.reshape(batch, self.signal_channels * self.cnn_N_filt[0], -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adapool(x)
        x = x.permute(0, 2, 1)
        tokens = self.token_proj(x)
        return tokens


class MultiModalResNet1D(nn.Module):
    def __init__(self, option_my, block_types, layers_dict, num_classes=1000,
                 token_len=32, token_dim=32, attn_heads=4):
        super(MultiModalResNet1D, self).__init__()
        self.branches = nn.ModuleDict()
        self.option_my = option_my
        for mod, cfg in option_my['modalities'].items():
            block = block_types[cfg['block']]
            layers = layers_dict[cfg['structure']]
            branch = SingleResNetBranch(cfg, block, layers, token_len, token_dim)
            self.branches[mod] = branch
        self.cross_attn = MultiheadAttention(embed_dim=token_dim, num_heads=attn_heads, batch_first=True)
        self.fc_fuse = nn.Linear(token_len * token_dim, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x_input):
        x_dict = {}
        idx = 0
        for mod, cfg in self.branches.items():
            ch = self.option_my['modalities'][mod]['channel_num']
            x_dict[mod] = x_input[:, idx:idx + ch, :]
            idx += ch

        tokens = []
        for mod, branch in self.branches.items():
            tokens_mod = branch(x_dict[mod])
            tokens.append(tokens_mod)
        
        batch = tokens[0].size(0)
        seq = torch.cat(tokens, dim=1)
        attn_out, _ = self.cross_attn(seq, seq, seq)
        
        t_len = tokens[0].size(1)
        num_mod = len(tokens)
        fused = attn_out.view(batch, num_mod, t_len, -1).mean(dim=1)
        fused_flat = fused.reshape(batch, -1)
        features = self.fc_fuse(fused_flat)
        logits = self.classifier(features)
        return logits, features, tokens

