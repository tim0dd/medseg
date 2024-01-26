from typing import List, Optional, Dict, Tuple

from beartype import beartype

from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType

CITYSCAPES_DEFAULT_PATH = "{project_root}/data/datasets/cityscapes"


class Cityscapes(MedsegDataset):

    def __init__(
            self,
            split_type: SplitType,
            dataset_path: str = CITYSCAPES_DEFAULT_PATH,
            use_custom_split: bool = False,
            custom_split_ratios: Optional[List[float]] = None,
            randomize_custom_split: bool = False,
            random_seed: int = 42,
            indices: Optional[List[int]] = None
    ):
        super().__init__(split_type, dataset_path, use_custom_split=use_custom_split,
                         custom_split_ratios=custom_split_ratios, randomize_custom_split=randomize_custom_split,
                         random_seed=random_seed, indices=indices)

    @beartype
    def get_class_colors(self) -> Optional[Dict[int, Tuple[int, int, int]]]:
        #  map trainId to color
        return {
            0: (128, 64, 128),
            1: (244, 35, 232),
            2: (70, 70, 70),
            3: (102, 102, 156),
            4: (190, 153, 153),
            5: (153, 153, 153),
            6: (250, 170, 30),
            7: (220, 220, 0),
            8: (107, 142, 35),
            9: (152, 251, 152),
            10: (70, 130, 180),
            11: (220, 20, 60),
            12: (255, 0, 0),
            13: (0, 0, 142),
            14: (0, 0, 70),
            15: (0, 60, 100),
            16: (0, 80, 100),
            17: (0, 0, 230),
            18: (119, 11, 32),
            19: (0, 0, 0),
            255: (0, 0, 0)
        }

    @beartype
    def get_train_id_to_regular_ids_mapping(self) -> Dict[int, int]:
        # map trainId to regularId
        return {
            255: 0,
            0: 7,
            1: 8,
            2: 11,
            3: 12,
            4: 13,
            5: 17,
            6: 19,
            7: 20,
            8: 21,
            9: 22,
            10: 23,
            11: 24,
            12: 25,
            13: 26,
            14: 27,
            15: 28,
            16: 31,
            17: 32,
            18: 33,
        }
