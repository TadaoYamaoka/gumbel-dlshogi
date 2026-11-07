from cshogi import HAND_PIECES, dlshogi
from cshogi.dlshogi import FEATURES1_NUM, make_move_label

__all__ = [
    "FEATURES1_NUM",
    "MOVE_PLANES_NUM",
    "MOVE_LABELS_NUM",
    "features2_num",
    "set_nyugyoku_features",
    "make_input_features",
    "make_move_label",
]

# 移動方向の数(8方向 + 桂馬の2方向、それぞれ成り・不成)
MOVE_DIRECTION_NUM = 20

# 移動を表すラベルの数
MOVE_PLANES_NUM = MOVE_DIRECTION_NUM + len(HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81


def set_nyugyoku_features(use: bool):
    """入玉特徴量を使うかを設定する。

    cshogi のプロセス単位の設定なので、特徴量を作成するプロセスごとに呼ぶ必要がある。
    """
    dlshogi.use_nyugyoku_features(use)


def features2_num() -> int:
    """入力特徴量2(持ち駒・王手・入玉)の数(入玉特徴量の設定に依存する)"""
    return dlshogi.FEATURES2_NUM


def make_input_features(board, features1, features2):
    """dlshogi 互換の入力特徴量を作成する。

    features1: `[FEATURES1_NUM, 9, 9]` の float32 の配列(駒の配置・利き・利き数)。
    features2: `[features2_num(), 9, 9]` の float32 の配列(持ち駒・王手・入玉)。
    """
    # cshogi は入玉特徴量の面をゼロクリアしないため、features2 全体をクリアしておく
    features2.fill(0)
    dlshogi.make_input_features(board, features1, features2)
