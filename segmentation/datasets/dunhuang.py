# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import os

from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset
from mmseg.datasets import PascalVOCDataset as _PascalVOCDataset


@DATASETS.register_module(force=True)
class DunhuangDataset20(CustomDataset):
    """Dunhuang dataset (the background class is ignored).
    Burrowed from MaskCLIP

    Args:
        split (str): Split txt file for Dunhuang.
    """

    # CLASSES = ('murals circular zigzag pattern', 'murals copper coin pattern',
    #            'murals honeysuckle pattern 1', 'murals flame pattern', 
    #            'murals honeysuckle pattern 2', 'murals bead pattern',
    #            'murals triangular canopy pattern', 'murals triangle pattern', 
    #            'murals bread-like pattern')
    
    CLASSES = ('circular zigzag pattern', 'copper coin pattern',
               'honeysuckle pattern 1', 'flame pattern', 
               'honeysuckle pattern 2', 'bead pattern',
               'triangular canopy pattern', 'murals triangle pattern', 
               'bread-like pattern')

    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0]]

    def __init__(self, **kwargs):
        super(DunhuangDataset20, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            # split=split,
            # reduce_zero_label=True,
            **kwargs)
        # assert os.path.exists(self.img_dir) and self.split is not None


@DATASETS.register_module(force=True)
class DunhuangDataset20WithBackground(CustomDataset):
    """Dunhuang dataset (the background class is ignored).
    Burrowed from MaskCLIP

    Args:
        split (str): Split txt file for Dunhuang.
    """

    CLASSES = (
        'background',

        'circular zigzag pattern', 'copper coin pattern', 'honeysuckle pattern 1',
        'flame pattern', 'honeysuckle pattern 2', 'linked bead pattern',
        'triangular canopy pattern', 'triangle pattern', 'bread-like pattern'
    )

    PALETTE = [
        [0, 0, 0],

        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0]
    ]

    def __init__(self, **kwargs):
        super(DunhuangDataset20WithBackground, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            # split=split,
            reduce_zero_label=False,
            **kwargs)
        # assert os.path.exists(self.img_dir) and self.split is not None
