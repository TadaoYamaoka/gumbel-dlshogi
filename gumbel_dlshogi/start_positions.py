"""Start positions of the self-play games.

A game starts from a position chosen at random from a file of start positions (the
same format as roots.hcp of dlshogi: HuffmanCodedPos records, e.g. made by
dlshogi/utils/csa_to_roots_hcp.py), or from the initial position (startpos).

The positions are not limited to those from the initial position (e.g. handicap
games), and a record is checked by Board.is_ok(), which prints the position to
stdout for a broken one.

HuffmanCodedPos has neither the move number nor the move history. So a game from
the file counts its moves from the start position (MAX_MOVES applies from there,
as dlshogi), and repetitions are detected only from the moves after it.
"""

import argparse
import math
import os

import numpy as np
from cshogi import Board, HuffmanCodedPos

# Sources of the start position of a game
STARTPOS = "startpos"
ROOTS = "roots"
SOURCES = (STARTPOS, ROOTS)

# Attempts to choose a valid start position from the file, before giving up
MAX_ATTEMPTS = 1000


def check_roots_file(path):
    """Raises ValueError if the file cannot be a file of start positions.

    The records themselves are checked when they are chosen (see StartPositions).
    """
    if not os.path.isfile(path):
        raise ValueError(f"The start positions file {path} is not found")
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError(f"The start positions file {path} is empty")
    if size % HuffmanCodedPos.itemsize:
        raise ValueError(
            f"The size of the start positions file {path} ({size} bytes) is not a "
            f"multiple of {HuffmanCodedPos.itemsize} bytes (HuffmanCodedPos)"
        )


def roots_ratio_type(value):
    """argparse type of roots_ratio: a finite number from 0 to 1."""
    try:
        ratio = float(value)
    except ValueError:
        ratio = math.nan
    if not (math.isfinite(ratio) and 0 <= ratio <= 1):
        raise argparse.ArgumentTypeError(
            f"roots_ratio must be a number from 0 to 1: {value!r}"
        )
    return ratio


def add_start_positions_arguments(parser):
    """Adds the arguments of the start positions of the self-play games."""
    parser.add_argument(
        "--roots",
        help="File of start positions (HuffmanCodedPos, the same format as "
        "roots.hcp of dlshogi). Each game starts from a position chosen at random.",
    )
    parser.add_argument(
        "--roots_ratio",
        type=roots_ratio_type,
        default=1.0,
        help="Ratio of the games started from --roots (the others start from the "
        "initial position). It is the ratio of the games, not of the positions.",
    )


def _has_valid_king_squares(record):
    """Whether the king squares encoded first in the record are on the board.

    Board.set_hcp() puts the kings without checking their squares, so a broken
    record may break the memory. The first bit is the side to move, and the next
    7 + 7 bits are the squares of the kings (from the lowest bit).
    """
    bits = int.from_bytes(record.tobytes()[:2], "little")
    black_king = (bits >> 1) & 0x7F
    white_king = (bits >> 8) & 0x7F
    return black_king < 81 and white_king < 81 and black_king != white_king


def _is_terminal(board):
    """Whether the game would end at once (the same rules as the self-play games)."""
    return board.is_game_over() or board.is_nyugyoku() or board.mate_move(3) != 0


class StartPositions:
    """Chooses the start position of each game.

    The file is opened by np.memmap (read only), so the processes share its pages.
    Open it in each process instead of passing the instance to the processes.
    """

    def __init__(self, roots=None, roots_ratio=1.0, rng=None):
        self.path = roots
        self.records = None
        if roots is not None:
            self.records = np.memmap(roots, dtype=HuffmanCodedPos, mode="r")
        self.roots_ratio = roots_ratio
        self.rng = np.random.default_rng() if rng is None else rng

    def set_start_position(self, board: Board) -> str:
        """Sets the start position of a new game to board, and returns its source.

        Raises:
            RuntimeError: If no valid start position is chosen in MAX_ATTEMPTS.
        """
        if self.records is None or self.rng.random() >= self.roots_ratio:
            board.reset()
            return STARTPOS

        broken = terminal = 0
        for _ in range(MAX_ATTEMPTS):
            index = self.rng.integers(len(self.records))
            record = np.array(self.records[index : index + 1])
            if not _has_valid_king_squares(record):
                broken += 1
                continue
            try:
                board.set_hcp(record)
            except RuntimeError:
                broken += 1
                continue
            # It prints the error and the position to stdout when it fails
            if not board.is_ok():
                broken += 1
                continue
            if _is_terminal(board):
                terminal += 1
                continue
            return ROOTS
        raise RuntimeError(
            f"No valid start position was chosen from {self.path} in {MAX_ATTEMPTS} "
            f"attempts ({broken} broken records, {terminal} positions where the game "
            "ends at once: game over, nyugyoku or mate in 3)"
        )
