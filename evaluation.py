from __future__ import annotations

from typing import Optional

import chess


MATE_SCORE = 100_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = [
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
]

EXTENDED_CENTER = [
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.D4,
    chess.E4,
    chess.F4,
    chess.C5,
    chess.D5,
    chess.E5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
]

WHITE_DEVELOPMENT_SQUARES = {
    chess.B1: chess.KNIGHT,
    chess.G1: chess.KNIGHT,
    chess.C1: chess.BISHOP,
    chess.F1: chess.BISHOP,
}

BLACK_DEVELOPMENT_SQUARES = {
    chess.B8: chess.KNIGHT,
    chess.G8: chess.KNIGHT,
    chess.C8: chess.BISHOP,
    chess.F8: chess.BISHOP,
}


def get_terminal_score(
    board: chess.Board,
    *,
    claim_draw: bool = True,
) -> Optional[int]:
    """Return the exact terminal score, or ``None`` if the game is ongoing.

    Scores are always expressed from White's perspective:
    - White win: ``+MATE_SCORE``
    - Black win: ``-MATE_SCORE``
    - Draw: ``0``

    When ``claim_draw`` is True, claimable draws such as the fifty-move rule
    and threefold repetition are treated as terminal outcomes.
    """

    outcome = board.outcome(claim_draw=claim_draw)

    if outcome is None:
        return None

    if outcome.winner is None:
        return 0

    return MATE_SCORE if outcome.winner == chess.WHITE else -MATE_SCORE


def evaluate_board(board: chess.Board, *, claim_draw: bool = True) -> int:
    """Evaluate the current chess position from White's perspective.

    Positive scores favour White and negative scores favour Black. Terminal
    positions use exact win/loss/draw values; non-terminal positions use the
    handcrafted heuristic components below.
    """

    terminal_score = get_terminal_score(board, claim_draw=claim_draw)
    if terminal_score is not None:
        return terminal_score

    score = 0
    score += evaluate_material(board)
    score += evaluate_piece_activity(board)
    score += evaluate_development(board)
    score += evaluate_center_control(board)
    score += evaluate_mobility(board)

    return score


def evaluate_material(board: chess.Board) -> int:
    """Evaluate material balance from White's perspective."""

    score = 0

    for piece_type, value in PIECE_VALUES.items():
        white_pieces = board.pieces(piece_type, chess.WHITE)
        black_pieces = board.pieces(piece_type, chess.BLACK)

        score += len(white_pieces) * value
        score -= len(black_pieces) * value

    return score


def evaluate_piece_activity(board: chess.Board) -> int:
    """Reward pieces closer to the centre and penalise edge knights."""

    score = 0

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        distance_from_center = abs(file_index - 3.5) + abs(rank_index - 3.5)
        activity_bonus = int(20 - 4 * distance_from_center)

        if piece.color == chess.WHITE:
            score += activity_bonus
        else:
            score -= activity_bonus

        if piece.piece_type == chess.KNIGHT:
            on_edge = (
                file_index == 0
                or file_index == 7
                or rank_index == 0
                or rank_index == 7
            )

            if on_edge:
                if piece.color == chess.WHITE:
                    score -= 20
                else:
                    score += 20

    return score


def evaluate_development(board: chess.Board) -> int:
    """Penalise undeveloped knights and bishops on their starting squares.

    This checks whether the original minor piece is still on its own starting
    square. It avoids rewarding a side merely because that square became empty
    after the piece was captured.
    """

    score = 0

    for square, expected_piece_type in WHITE_DEVELOPMENT_SQUARES.items():
        piece = board.piece_at(square)
        if (
            piece is not None
            and piece.color == chess.WHITE
            and piece.piece_type == expected_piece_type
        ):
            score -= 10

    for square, expected_piece_type in BLACK_DEVELOPMENT_SQUARES.items():
        piece = board.piece_at(square)
        if (
            piece is not None
            and piece.color == chess.BLACK
            and piece.piece_type == expected_piece_type
        ):
            score += 10

    return score


def evaluate_center_control(board: chess.Board) -> int:
    """Reward occupying or attacking central squares."""

    score = 0

    for square in CENTER_SQUARES:
        piece = board.piece_at(square)

        if piece is not None:
            if piece.color == chess.WHITE:
                score += 15
            else:
                score -= 15

        white_attackers = board.attackers(chess.WHITE, square)
        black_attackers = board.attackers(chess.BLACK, square)

        score += len(white_attackers) * 5
        score -= len(black_attackers) * 5

    for square in EXTENDED_CENTER:
        piece = board.piece_at(square)

        if piece is not None:
            if piece.color == chess.WHITE:
                score += 5
            else:
                score -= 5

    return score


def evaluate_mobility(board: chess.Board) -> int:
    """Estimate mobility as the difference in legal-move counts."""

    white_board = board.copy(stack=False)
    white_board.turn = chess.WHITE

    black_board = board.copy(stack=False)
    black_board.turn = chess.BLACK

    white_mobility = len(list(white_board.legal_moves))
    black_mobility = len(list(black_board.legal_moves))

    return 2 * (white_mobility - black_mobility)
