import numpy as np
from cshogi import HuffmanCodedPos
from pydlshogi2.features import MOVE_LABELS_NUM

dtypeTrainingData = np.dtype(
    [
        ("hcp", HuffmanCodedPos),
        ("policy", np.dtype((np.float32, MOVE_LABELS_NUM))),
        ("result", np.uint8),
    ]
)
