import os
from collections import OrderedDict
from typing import Callable, List

import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.nn import functional as F

from medseg.data.split_type import SplitType
from medseg.models.segmentors.segmentor import Segmentor
from medseg.training.loss.hardnet_dfus_loss import HarDNetDFUSBoundaryLoss, HarDNetDFUSLoss
from medseg.util.img_ops import logits_to_segmentation_mask, multiscale
from medseg.util.path_builder import PathBuilder
from .components import Flatten, ConvLayer, DWConvLayer, HarDBlockV2, SELayer
from .lawin_decoder import LawinHead5
from .model_ema import HarDNetModelEMA


# Adapted from https://github.com/YuWenLo/HarDNet-DFUS
class HarDNetDFUS(Segmentor):
    """
    HarDNetV2-53-Lawin (aka HarDNet-DFUS)
    """

    def __init__(self, out_channels=1, in_size=512, dropout=0.2, use_dw_conv=False, use_ema=True,
                 deep_supervision=True):
        super().__init__()
        self.in_size = in_size
        self.out_channels = out_channels
        #TODO: adapt for multiclass
        self.boundary_loss = HarDNetDFUSBoundaryLoss()
        self.deep_supervision = deep_supervision
        self.backbone = HarDNet53(use_dw_conv=use_dw_conv, pretrained=True, dropout=dropout)
        bb_out_ch = [140, 540, 800, 1200]
        self.head = LawinHead5(in_channels=bb_out_ch, num_classes=out_channels)
        self.last3_seg = nn.Conv2d(512, out_channels, kernel_size=1)
        self.last3_seg2 = nn.Conv2d(768, out_channels, kernel_size=1)
        self.use_ema = use_ema
        self.ema = HarDNetModelEMA(self)

    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)[-4:]

        output, last3_feat, last3_feat2, low_level_feat = self.head(x_4, x_8, x_16, x_32)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')

        if self.training:
            last3_feat = F.interpolate(self.last3_seg(last3_feat), size=x.size()[2:], mode='bilinear')
            last3_feat2 = F.interpolate(self.last3_seg2(last3_feat2), size=x.size()[2:], mode='bilinear')
            low_level_feat = F.interpolate(low_level_feat, size=x.size()[2:], mode='bilinear')
            return output, last3_feat, last3_feat2, low_level_feat

        return output

    def default_loss_func(self, multiclass: bool) -> Callable[[Tensor, Tensor], Tensor]:
        # TODO test if this works for multiclass
        return HarDNetDFUSLoss()

    def set_param_groups(self, optimizer: torch.optim.Optimizer):
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.named_parameters()).items():
            if '.bias' in k:
                pg2.append(v)  # biases
            elif 'Conv2d.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'conv.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'position_mixing' in k:
                pg1.append(v)  # apply weight_decay
            else:
                pg0.append(v)  # all else

        optimizer.param_groups = []  # remove default parameter groups
        optimizer.add_param_group({'params': pg0})  # add pg0 with weight_decay
        is_adamw = isinstance(optimizer, torch.optim.AdamW)
        optimizer.add_param_group(
            {'params': pg1, 'weight_decay': 0.01 if is_adamw else 0.0005})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        del pg0, pg1, pg2

    def train_iteration(self, images: Tensor, masks: Tensor, ids: List[int], state):

        with autocast(enabled=state.mixed_precision):
            state.current_iteration += 1
            images = images.to(device=state.device, dtype=torch.float)
            masks = masks.to(device=state.device, dtype=torch.long if state.multiclass else torch.float)

            if self.multiscale_cfg is not None:
                images = multiscale(self.in_size, images, masks, **self.multiscale_cfg)
                masks = multiscale(self.in_size, images, masks, **self.multiscale_cfg)

            # forward
            hardnet_output = state.compiled_model(images)
            masks = masks.squeeze(1) if state.multiclass else masks
            loss = state.loss_func(hardnet_output[0], masks)
            if self.deep_supervision:
                deep_loss = state.loss_func(hardnet_output[1], masks)
                deep_loss2 = state.loss_func(hardnet_output[2], masks)
                boundary_loss = self.boundary_loss(hardnet_output[3], masks)

            # backward
            state.optimizer.zero_grad()
            state.scaler.scale(loss).backward(retain_graph=True)
            if self.deep_supervision:
                state.scaler.scale(deep_loss).backward(retain_graph=True)
                state.scaler.scale(deep_loss2).backward(retain_graph=True)
                state.scaler.scale(boundary_loss).backward()
            state.scaler.step(state.optimizer)
            state.scaler.update()
            if self.use_ema:
                self.ema.update(self)
                self.ema.update_attr(self)
            state.train_loss_tracker.update(loss)

            # metrics
            predictions = hardnet_output[0]
            predictions = logits_to_segmentation_mask(predictions).long() if state.multiclass else predictions > 0.5
            img_ids = [state.dataset_manager.get_train_dataset().get_image_file_name(real_i) for real_i in ids]
            train_metrics_tracker = state.metrics_manager.get_last_tracker(SplitType.TRAIN)
            train_metrics_tracker.update_metrics_from_batch(img_ids, predictions.cpu(), masks.cpu())


# %%
class HarDNet53(nn.Module):
    """
    The HarDNet backbone, which is named KingNet53 in their code but not mentioned by name in the different HarDNet
    papers.
    """

    def __init__(self,
                 use_dw_conv=False,
                 pretrained=False,
                 first_ch=[30, 60],
                 ch_list=[120, 240, 540, 800, 1200],
                 n_layers=[9, 9, 15, 9, 3],
                 downSamp=[1, 0, 1, 1, 0],
                 dropout=0.2):
        super().__init__()

        max_pool = not use_dw_conv
        # if use_dw_conv:
        #     max_pool = False
        # drop_rate = 0.05

        blocks = len(n_layers)

        self.base = nn.ModuleList([])
        # Stem Layer
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, bias=False))
        self.base.append(ConvLayer(in_channels=first_ch[0], out_channels=first_ch[1], kernel=3))

        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all KingNet blocks
        ch = first_ch[1]
        for i in range(blocks):
            blk = HarDBlockV2(ch, n_layers[i], dwconv=use_dw_conv)
            out_ch = blk.get_out_ch()
            self.base.append(blk)
            self.base.append(SELayer(out_ch))
            self.base.append(ConvLayer(out_ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blocks - 1]
        self.base.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(dropout),
                nn.Linear(ch, 1000))
        )

        if pretrained:
            weight_file = PathBuilder.pretrained_dir_builder().add('kingnet53.pth').build()
            if not os.path.isfile(weight_file):
                print(weight_file, 'is not found')
                exit(0)
            weights = torch.load(weight_file)
            state_dict = weights["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # remove 'module' prefix from the keys
                name = k[7:]
                if name in self.state_dict():
                    new_state_dict[name] = v
            # load params
            self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        out_branch = []

        for i in range(len(self.base) - 1):

            x = self.base[i](x)
            # 0-9:30,60,60,140,140,120,120,280,280,240
            # 10-20:720,720,540,540,1260,1260,800,800,1200,1200,1200
            if i == 4 or i == 12 or i == 16 or i == 20:
                out_branch.append(x)

        return out_branch
