import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, so the export and its check build the same weights from the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
