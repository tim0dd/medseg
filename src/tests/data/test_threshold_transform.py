import numpy as np
import pytest
import torch
from PIL import Image
from PIL.Image import Image as PILImage

from medseg.data.transforms.transforms import Threshold


def generate_test_data(size=(100, 100)):
    return np.random.rand(*size)


@pytest.mark.parametrize("input_type", [PILImage, np.ndarray, torch.Tensor])
@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("pixel_min, pixel_max", [(0, 1), (0, 255), (-1, 1)])
@pytest.mark.parametrize("backend", ['albumentations', 'torchvision'])
def test_threshold(input_type, threshold, pixel_min, pixel_max, backend):
    test_data = generate_test_data()
    if input_type == Image:
        test_input = Image.fromarray((test_data * 255).astype(np.uint8))
    elif input_type == np.ndarray:
        test_input = test_data
    else:
        test_input = torch.from_numpy(test_data)

    args = {"threshold": threshold, "pixel_min": pixel_min, "pixel_max": pixel_max}
    transform = Threshold(args, backend)

    transformed_data = transform.img_transforms(test_input)

    if isinstance(transformed_data, PILImage):
        transformed_data = np.array(transformed_data) / 255
    elif isinstance(transformed_data, torch.Tensor):
        transformed_data = transformed_data.numpy()

    expected_data = np.where(test_data >= threshold, pixel_max, pixel_min)
    assert np.allclose(transformed_data, expected_data, rtol=1e-5, atol=1e-8)
