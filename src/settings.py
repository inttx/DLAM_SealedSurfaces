import warnings

import torch.cuda
from rasterio.errors import NotGeoreferencedWarning

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed=42)

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
