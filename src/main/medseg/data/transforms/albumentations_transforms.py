import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2 as AToTensorV2
from torch import Tensor

from medseg.data.transforms.transforms import MedsegTransform, MedsegAlbumentationsTransform


class OneOf(MedsegAlbumentationsTransform):
    def __init__(self, args: dict, nested_transforms: list):
        if nested_transforms is None or not isinstance(nested_transforms, list) or len(nested_transforms) < 2:
            raise ValueError("AOneOf requires at least 2 transforms to choose from")
        for nested_transform in nested_transforms:
            if not isinstance(nested_transform, MedsegTransform):
                raise ValueError("Found an invalid item in nested transforms list")
        self.nested_transforms = nested_transforms
        super().__init__(A.OneOf, args)

    def __eq__(self, other):
        is_equal = super().__eq__(other)
        if isinstance(other, OneOf):
            for i, nested_transform in enumerate(self.nested_transforms):
                if i >= len(other.nested_transforms):
                    return False
                is_equal = is_equal and nested_transform.__eq__(other.nested_transforms[i])
        return is_equal

    def compose(self):
        composed_transforms = []
        for nested_transform in self.nested_transforms:
            nested_transform.compose()
            composed_transforms.append(nested_transform.img_transforms)
        self.args['transforms'] = composed_transforms
        super().compose()


class CenterCrop(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.CenterCrop, args)


class RandomCrop(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.RandomCrop, args)


class HorizontalFlip(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.HorizontalFlip, args)


class VerticalFlip(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.VerticalFlip, args)


class ShiftScaleRotate(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.ShiftScaleRotate, args)


class CoarseDropout(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.CoarseDropout, args)


class GaussNoise(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.GaussNoise, args)


class ToTensorV2(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(AToTensorV2, args)


class PadIfNeeded(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.PadIfNeeded, args)


class ColorJitter(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.ColorJitter, args)


class Resize(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.Resize, args)


class Normalize(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.Normalize, args)

    def denormalize(self, img: Tensor, mask: Tensor):
        # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
        mean = self.args['mean']
        std = self.args['std']
        mean = np.array(mean)
        std = np.array(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        temp_args = {'mean': mean_inv.tolist(), 'std': std_inv.tolist()}
        return self.apply(img.numpy(), mask.numpy(), temp_args=temp_args)


class SquarePad(MedsegAlbumentationsTransform):
    def __init__(self, args: dict):
        super().__init__(A.PadIfNeeded, args)
        self.needs_height_width = True

    def compose(self):
        self.args['border_mode'] = cv2.BORDER_CONSTANT
        if 'h' in self.args and 'w' in self.args:
            square_size = max(self.args['h'], self.args['w'])
            self.args['min_width'] = square_size
            self.args['min_height'] = square_size
        if 'h' in self.args:
            del self.args['h']
        if 'w' in self.args:
            del self.args['w']
        super().compose()
