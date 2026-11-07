import numpy as np
from cshogi import HuffmanCodedPos

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
