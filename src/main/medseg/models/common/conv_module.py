from torch import nn, Tensor


class ConvModule(nn.Module):
    # incomplete reconstruction of ConvModule from mmcv (only encompasses required parts) for the purpose of
    # replacing it in different models
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True, conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        if norm_cfg is not None and 'type' in norm_cfg:
            if norm_cfg['type'].lower() == 'gn':
                num_groups = act_cfg.get('num_groups', 32)
                self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.act = nn.Identity()
        if act_cfg is not None and 'type' in act_cfg:
            if act_cfg['type'].lower() == 'relu':
                in_place = act_cfg.get('in_place', True)
                self.act = nn.ReLU(in_place)

        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))
