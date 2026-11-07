import os
from pathlib import Path

import numpy as np
from cshogi import HuffmanCodedPos


def save_atomically(path, save):
    """Saves with save(temporary_path) and renames it to path.

    The path never holds a partially written file, so its existence shows that the
    save has completed even if the process stops during the save.
    """
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        save(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


# 学習データは局面ファイルと方策ファイルの2つで1組になる。
# 方策は確率が0でない手だけを保存する(非合法手は必ず0、合法手もアンダーフローで0になり得る)。

# 局面ファイル(固定長): 局面、対局結果、方策ファイルに保存した手の数
POSITION_FILE_EXT = ".pos"
dtypePosition = np.dtype(
    [
        ("hcp", HuffmanCodedPos),
        ("result", np.uint8),
        ("num_moves", np.uint16),
    ]
)

# 方策ファイル(可変長): 局面の順に、手のラベルと確率を num_moves 個ずつ並べる
POLICY_FILE_EXT = ".policy"
dtypeMovePolicy = np.dtype(
    [
        ("label", np.uint16),
        ("prob", np.float32),
    ]
)
