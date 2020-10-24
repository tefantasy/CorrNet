import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCorrelationBlock(nn.Module):
    """
    A separate block added at the end of each res-stage in R(2+1)D. 
    See the paper. 
    """
    def __init__(self, num_channel, seq_len, filter_size, dilation=1, num_groups=1, mode="sum"):
        """
        Args:
            see WeightedCorrelationLayer.
        """
        super(WeightedCorrelationBlock, self).__init__()

        assert mode in ["sum", "concat"]
        assert num_channel % 4 == 0

        self.encode_conv = nn.Sequential(
            nn.Conv3d(num_channel, num_channel // 4, 1, bias=False),
            nn.BatchNorm3d(num_channel // 4), 
            nn.ReLU(inplace=True)
        )

        self.correlation = WeightedCorrelationLayer(
            num_channel // 4, seq_len, filter_size, dilation, num_groups
        )
        if mode == "concat":
            assert num_channel > filter_size * filter_size * num_groups
            self.sum_mode = False

            self.bypass_conv = nn.Sequential(
                nn.Conv3d(
                    num_channel // 4, num_channel - filter_size * filter_size * num_groups, 1, bias=False
                ),
                nn.BatchNorm3d(num_channel - filter_size * filter_size * num_groups)
            )

        elif mode == "sum":
            self.sum_mode = True
            self.decode_conv = nn.Sequential(
                nn.Conv3d(filter_size * filter_size * num_groups, num_channel, 1, bias=False),
                nn.BatchNorm3d(num_channel)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channel, seq, h, w)
        Returns:
            out (Tensor): shape (batch, channel, seq, h, w)
        """
        x_enc = self.encode_conv(x)
        x_corr = self.correlation(x_enc)

        if self.sum_mode:
            x_dec = self.decode_conv(x_corr)
            out = x + x_dec
        else:
            x_bypass = self.bypass_conv(x_enc)
            out = torch.cat([x_corr, x_bypass], 1)
        out = self.relu(out)
        return out


class WeightedCorrelationLayer(nn.Module):
    """
    Weighted Correlation Layer proposed in paper
    ``Video Modeling with Correlation Networks``. 
    """
    def __init__(self, in_channel, seq_len, filter_size, dilation=1, num_groups=1):
        """
        Args:
            in_channel: C
            seq_len: L
            filter_size: K
            dilation: D. If greater than 1, perform dilated correlation.
            num_groups: G. If greater than 1, perform groupwise correlation.
        """
        super(WeightedCorrelationLayer, self).__init__()

        assert dilation >= 1, "Dilation must be greater than 1. "
        assert num_groups >= 1, "Group number must be greater than 1. "
        assert filter_size % 2 == 1, "Only support odd K. "
        assert in_channel % num_groups == 0, "Group number must be a divisor of channel number. "

        self.filter_weight = nn.Parameter(torch.Tensor(in_channel, seq_len, filter_size, filter_size))
        nn.init.kaiming_normal_(self.filter_weight, mode='fan_out', nonlinearity='relu')

        self.in_channel = in_channel
        self.seq_len = seq_len
        self.dilation = dilation
        self.num_groups = num_groups
        self.span_size = (filter_size - 1) * dilation + 1
        self.pad_size = (self.span_size - 1) // 2

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channel, seq, h, w)
        Returns:
            flatten_response (Tensor): shape (batch, n_groups*k^2, seq, h, w)
        """

        # second image in each correlation operation
        x2 = F.pad(
            x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            'constant', 0
        )
        # first image in each correlation operation
        # repeat the first frame once to perform self-correlation
        x1 = torch.cat(
            (x[:, :, [0], :, :], x[:, :, :-1, :, :]), 2
        )

        offset_y, offset_x = torch.meshgrid(
            torch.arange(0, self.span_size, self.dilation), 
            torch.arange(0, self.span_size, self.dilation)
        )

        batch_size, c, t, h, w = x.size()
        position_response_list = []

        # for each position in the filter, calculate all responses between two frames
        for dx, dy in zip(offset_x.reshape(-1), offset_y.reshape(-1)):
            pos_filter_weight = self.filter_weight[:, :, dy//self.dilation, dx//self.dilation].view(1, c, t, 1, 1).expand(
                batch_size, -1, -1, h, w
            )
            position_response = pos_filter_weight * x1 * x2[:, :, :, dy:dy+h, dx:dx+w]

            # perform groupwise mean
            position_response = position_response.reshape(
                -1, self.num_groups, c // self.num_groups, t, h, w
            )
            position_response = torch.mean(position_response, 2)

            # position_response: (batch, n_groups, t, h, w)
            position_response_list.append(position_response)

        flatten_response = torch.cat(position_response_list, 1)
        return flatten_response
