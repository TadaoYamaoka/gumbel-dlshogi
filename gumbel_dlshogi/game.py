"""The end of a game, shared by the search, self-play and the evaluation of models."""

from typing import Optional

from cshogi import BLACK, BLACK_WIN, DRAW, REPETITION_DRAW, REPETITION_LOSE
from cshogi import REPETITION_WIN, WHITE_WIN, Board

# A game is a draw at this move number (counted from the start position)
MAX_MOVES = 512

# How a game ends, in the order of the checks of end_reason, with the value (the
# game result) for the side to move at the end
END_REASONS = {
    "checkmate": 0.0,  # the side to move has no legal moves
    "nyugyoku": 1.0,  # the side to move can declare (入玉宣言)
    "repetition_draw": 0.5,  # 千日手
    "repetition_win": 1.0,  # 連続王手の千日手で手番側の勝ち
    "repetition_lose": 0.0,  # 連続王手の千日手で手番側の負け
    "max_moves": 0.5,
    "mate_in_3ply": 1.0,  # the side to move has a mate in 3 plies (the game ends)
}

# Names of the game results in the records of the games
RESULT_NAMES = {BLACK_WIN: "black_win", WHITE_WIN: "white_win", DRAW: "draw"}


def end_reason(board: Board) -> Optional[str]:
    """Returns how the game ends at the position (one of END_REASONS), or None."""
    if board.is_game_over():
        return "checkmate"
    if board.is_nyugyoku():
        return "nyugyoku"
    draw = board.is_draw()
    if draw == REPETITION_DRAW:
        return "repetition_draw"
    if draw == REPETITION_WIN:
        return "repetition_win"
    if draw == REPETITION_LOSE:
        return "repetition_lose"
    if board.move_number >= MAX_MOVES:
        return "max_moves"
    if board.mate_move(3) != 0:
        return "mate_in_3ply"
    return None


def game_result(reason: str, turn: int) -> int:
    """Returns the result (BLACK_WIN, WHITE_WIN or DRAW) of the game ended for the
    side to move `turn` at the end."""
    value = END_REASONS[reason]
    if value == 0.5:
        return DRAW
    side_to_move_wins = value == 1.0
    if side_to_move_wins == (turn == BLACK):
        return BLACK_WIN
    return WHITE_WIN
