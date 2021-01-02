import torch
import torch.nn as nn

from .conv_2dplus1d import Conv2DPlus1D
from .corr import WeightedCorrelationBlock
from ..build import MODEL_REGISTRY


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size, stride=1):

        super(Bottleneck, self).__init__()

        padding = kernel_size // 2

        self.downsample = (stride > 1) if isinstance(stride, int) else (max(stride) > 1)
        self.stride = stride

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 2+1D Conv
        self.conv2 = nn.Sequential(
            Conv2DPlus1D(planes, planes, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample_conv(x)

        out += residual
        out = self.relu(out)

        return out


class CorrNetImpl(nn.Module):

    def __init__(self, layer_sizes, corr_block_locs, num_frames, num_classes=400):
        """
        Args:
            layer_sizes (Seq[int]): number of blocks per layer (including correlation blocks). 
            corr_block_locs (Seq[Seq[int]/None]): 
                indices of correlation blocks in each layer. None if no correlation block is inserted.
            num_classes (int): number of output classes. 
        """
        super(CorrNetImpl, self).__init__()

        self.num_frames = num_frames

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.res2 = self._make_layer(
            32, 64, 3, layer_sizes[0], corr_block_locs[0], stride=(1, 2, 2)
        )
        self.res3 = self._make_layer(
            64 * Bottleneck.expansion, 128, 3, layer_sizes[1], corr_block_locs[1], stride=(1, 2, 2)
        )
        self.res4 = self._make_layer(
            128 * Bottleneck.expansion, 256, 3, layer_sizes[2], corr_block_locs[2], stride=(2, 2, 2)
        )
        self.res5 = self._make_layer(
            256 * Bottleneck.expansion, 512, 3, layer_sizes[3], corr_block_locs[3], stride=(2, 2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._initialize_weights()

    def forward(self, x):
        """
        Args:
            x (list): Contains exactly ONE tensor of shape (b, c, t, h, w). 
                      This follows the interface of PySlowFast. 
        Returns:
            x (Tensor): Logits for each class. 
        """
        x = x[0]
        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, inplanes, planes, kernel_size, layer_size, corr_block_locs, stride=1):
        if corr_block_locs:
            assert min(corr_block_locs) > 0, "Can not insert correlation block at the first location of a layer."
        else:
            corr_block_locs = []
        
        blocks = []

        # first block, need downsampling if stride > 1
        blocks.append(Bottleneck(inplanes, planes, kernel_size, stride))

        # decrease temporal length
        temporal_stride = stride if isinstance(stride, int) else stride[0]
        self.num_frames = self.num_frames // temporal_stride

        inplanes = planes * Bottleneck.expansion
        for i in range(1, layer_size):
            if i in corr_block_locs:
                blocks.append(
                    WeightedCorrelationBlock(inplanes, self.num_frames, filter_size=7, dilation=2, num_groups=32)
                )
            else:
                # no downsampling
                blocks.append(Bottleneck(inplanes, planes, kernel_size))

        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

@MODEL_REGISTRY.register()
class CorrNet(CorrNetImpl):

    def __init__(self, cfg):
        assert cfg.CORR_NET.TYPE in ["corr_26", "corr_50", "corr_101"]
        
        if cfg.CORR_NET.TYPE == "corr_26":
            super(CorrNet, self).__init__(
                [3, 3, 3, 2], [[2], [2], [2], None], cfg.DATA.NUM_FRAMES, cfg.MODEL.NUM_CLASSES
            )
        elif cfg.CORR_NET.TYPE == "corr_50":
            super(CorrNet, self).__init__(
                [4, 5, 7, 3], [[3], [4], [6], None], cfg.DATA.NUM_FRAMES, cfg.MODEL.NUM_CLASSES
            )
        elif cfg.CORR_NET.TYPE == "corr_101":
            super(CorrNet, self).__init__(
                [4, 5, 25, 3], [[3], [4], [12, 24], None], cfg.DATA.NUM_FRAMES, cfg.MODEL.NUM_CLASSES
            )
