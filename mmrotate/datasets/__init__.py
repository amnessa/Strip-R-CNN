# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .dota_1_5 import DOTADataset15, DOTAv15Dataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .fair import FairDataset
from .dior import DIORDataset

__all__ = [
	'SARDataset', 'DIORDataset', 'DOTADataset', 'DOTADataset15',
	'DOTAv15Dataset', 'build_dataset', 'HRSCDataset', 'FairDataset'
]
