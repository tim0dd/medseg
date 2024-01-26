import numpy as np
import pytest

from medseg.data.transforms.albumentations_transforms import SquarePad


@pytest.fixture
def non_square_image():
    return np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)


@pytest.fixture
def non_square_mask():
    return np.random.randint(0, 2, (100, 150), dtype=np.uint8)


def test_square_pad_non_square_input(non_square_image, non_square_mask):
    args = {'h': 200, 'w': 200}
    square_pad = SquarePad(args)
    square_pad.compose()

    img_padded, mask_padded = square_pad.apply(non_square_image, non_square_mask)

    assert img_padded.shape == (200, 200, 3)
    assert mask_padded.shape == (200, 200)
