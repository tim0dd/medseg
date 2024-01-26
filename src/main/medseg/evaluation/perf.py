import time

import torch
from torch import nn


# TODO: unused
# from https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/utils/utils.py (MIT License)
def throughput(dataloader, model: nn.Module, times: int = 30):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.cuda(non_blocking=True)
    B = images.shape[0]
    print(f"Throughput averaged with {times} times")
    start = time_sync()
    for _ in range(times):
        model(images)
    end = time_sync()

    print(f"Batch Size {B} throughput {times * B / (end - start)} images/s")


# from https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/utils/utils.py (MIT License)
def time_sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# TODO: use fvcore by FAIR to measure FLOPs
