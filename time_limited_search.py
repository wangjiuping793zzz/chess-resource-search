"""Time-limited ordered Alpha-Beta search and volatility-based allocation.

This module is intentionally separate from ``search.py`` so that the frozen
fixed-depth implementation remains unchanged.  The evaluator is shared with
the earlier experiments.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import chess

from evaluation import evaluate_board


class SearchTimeout(Exception):
    """Raised internally when a search reaches its absolute deadline."""


VOLATILITY_WEIGHTS = {
    "in_check": 0.35,
    "captures": 0.25,
    "checking_moves": 0.20,
    "evaluation_swing": 0.20,
}

QUIET_THRESHOLD = 0.15
VOLATILE_THRESHOLD = 0.35
TIME_MULTIPLIERS = {
    "quiet": 0.60,
    "normal": 1.00,
    "volatile": 1.80,
}

# Preserve at least the quiet-policy allocation for every nominal future move.
# The earlier 0.25 reserve could leave only 12.5 ms at the 50 ms budget, which
# was insufficient to complete depth 1 in high-branching volatile positions.
FUTURE_RESERVE_FACTOR = TIME_MULTIPLIERS["quiet"]


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise SearchTimeout


def _ordered_moves(board: chess.Board, deadline: float) -> List[chess.Move]:
    """Match the original move-ordering priorities, with deadline checks."""

    scored_moves = []
    for move_index, move in enumerate(board.legal_moves):
        _check_deadline(deadline)
        score = 0

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            attacking_piece = board.piece_at(move.from_square)
            if captured_piece is not None and attacking_piece is not None:
                score += 10 * captured_piece.piece_type - attacking_piece.piece_type
            else:
                # Covers en passant, where the destination square is empty.
                score += 10

        if move.promotion is not None:
            score += 100

        if board.gives_check(move):
            score += 50

        # Python sorting is stable.  Retaining the original move index makes
        # the deterministic tie-breaking rule explicit in the output logic.
        scored_moves.append((score, -move_index, move))

    scored_moves.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored_moves]


def _alpha_beta_timed(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    deadline: float,
    stats: Dict[str, int],
) -> int:
    _check_deadline(deadline)
    stats["nodes"] += 1

    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    moves = _ordered_moves(board, deadline)

    if maximizing_player:
        best_score = -float("inf")
        for move in moves:
            _check_deadline(deadline)
            board.push(move)
            try:
                score = _alpha_beta_timed(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    deadline,
                    stats,
                )
            finally:
                board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                stats["cutoffs"] += 1
                break
        return int(best_score)

    best_score = float("inf")
    for move in moves:
        _check_deadline(deadline)
        board.push(move)
        try:
            score = _alpha_beta_timed(
                board,
                depth - 1,
                alpha,
                beta,
                True,
                deadline,
                stats,
            )
        finally:
            board.pop()

        best_score = min(best_score, score)
        beta = min(beta, best_score)
        if beta <= alpha:
            stats["cutoffs"] += 1
            break
    return int(best_score)


def _search_one_depth(
    board: chess.Board,
    depth: int,
    deadline: float,
    stats: Dict[str, int],
) -> Tuple[Optional[chess.Move], int]:
    """Complete one ordered Alpha-Beta iteration at an exact depth."""

    _check_deadline(deadline)
    moves = _ordered_moves(board, deadline)
    if not moves:
        return None, evaluate_board(board)

    best_move = None
    alpha = -float("inf")
    beta = float("inf")

    if board.turn == chess.WHITE:
        best_score = -float("inf")
        for move in moves:
            _check_deadline(deadline)
            board.push(move)
            try:
                score = _alpha_beta_timed(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    deadline,
                    stats,
                )
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        return best_move, int(best_score)

    best_score = float("inf")
    for move in moves:
        _check_deadline(deadline)
        board.push(move)
        try:
            score = _alpha_beta_timed(
                board,
                depth - 1,
                alpha,
                beta,
                True,
                deadline,
                stats,
            )
        finally:
            board.pop()

        if score < best_score:
            best_score = score
            best_move = move
        beta = min(beta, best_score)
    return best_move, int(best_score)


def find_best_move_iterative(
    board: chess.Board,
    deadline: float,
    max_depth: int = 8,
) -> Tuple[Optional[chess.Move], Optional[int], Dict[str, object]]:
    """Search successively deeper and keep the last completed result.

    Work from an interrupted iteration is counted in ``nodes`` and ``cutoffs``
    but its move is discarded.  If even depth 1 cannot finish, the first legal
    move is returned as a deterministic depth-0 fallback.
    """

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, evaluate_board(board), {
            "nodes": 0,
            "cutoffs": 0,
            "completed_depth": 0,
            "attempted_depth": 0,
            "timed_out": False,
            "completed_iterations": 0,
            "iteration_nodes": [],
        }

    best_move = legal_moves[0]
    best_score = None
    completed_depth = 0
    attempted_depth = 0
    timed_out = False
    stats: Dict[str, int] = {"nodes": 0, "cutoffs": 0}
    iteration_nodes = []

    for depth in range(1, max_depth + 1):
        attempted_depth = depth
        nodes_before = stats["nodes"]
        try:
            move, score = _search_one_depth(board, depth, deadline, stats)
        except SearchTimeout:
            timed_out = True
            break

        if move is not None:
            best_move = move
            best_score = score
        completed_depth = depth
        iteration_nodes.append(stats["nodes"] - nodes_before)

    return best_move, best_score, {
        "nodes": stats["nodes"],
        "cutoffs": stats["cutoffs"],
        "completed_depth": completed_depth,
        "attempted_depth": attempted_depth,
        "timed_out": timed_out,
        "completed_iterations": completed_depth,
        "iteration_nodes": iteration_nodes,
    }


def measure_position_volatility(board: chess.Board) -> Dict[str, object]:
    """Return an interpretable 0--1 tactical-volatility score.

    The score deliberately excludes raw legal-move count: branching factor is
    logged, but a position with many quiet alternatives is not automatically
    considered tactically volatile.
    """

    legal_moves = list(board.legal_moves)
    capture_count = sum(board.is_capture(move) for move in legal_moves)
    checking_move_count = sum(board.gives_check(move) for move in legal_moves)
    in_check = board.is_check()

    current_evaluation = evaluate_board(board)
    previous_evaluation = current_evaluation
    if board.move_stack:
        previous_board = board.copy(stack=True)
        previous_board.pop()
        previous_evaluation = evaluate_board(previous_board)

    evaluation_swing = abs(current_evaluation - previous_evaluation)
    capture_component = min(capture_count / 3.0, 1.0)
    checking_component = min(checking_move_count / 2.0, 1.0)
    swing_component = min(evaluation_swing / 200.0, 1.0)

    volatility_score = (
        VOLATILITY_WEIGHTS["in_check"] * int(in_check)
        + VOLATILITY_WEIGHTS["captures"] * capture_component
        + VOLATILITY_WEIGHTS["checking_moves"] * checking_component
        + VOLATILITY_WEIGHTS["evaluation_swing"] * swing_component
    )

    if volatility_score < QUIET_THRESHOLD:
        volatility_class = "quiet"
    elif volatility_score < VOLATILE_THRESHOLD:
        volatility_class = "normal"
    else:
        volatility_class = "volatile"

    return {
        "volatility_score": volatility_score,
        "volatility_class": volatility_class,
        "time_multiplier": TIME_MULTIPLIERS[volatility_class],
        "in_check": in_check,
        "legal_move_count": len(legal_moves),
        "capture_move_count": capture_count,
        "checking_move_count": checking_move_count,
        "current_static_evaluation": current_evaluation,
        "previous_static_evaluation": previous_evaluation,
        "absolute_evaluation_swing": evaluation_swing,
    }


def allocate_decision_time(
    strategy: str,
    base_time_per_move: float,
    remaining_clock: float,
    remaining_move_capacity: int,
    volatility: Optional[Dict[str, object]] = None,
    future_reserve_factor: float = FUTURE_RESERVE_FACTOR,
) -> float:
    """Allocate time while preserving the quiet-policy budget for later moves."""

    if strategy not in {"fixed", "dynamic"}:
        raise ValueError("strategy must be 'fixed' or 'dynamic'")
    if base_time_per_move <= 0:
        raise ValueError("base_time_per_move must be positive")
    if remaining_clock <= 0:
        return 0.0

    if strategy == "fixed":
        return min(base_time_per_move, remaining_clock)

    if volatility is None:
        raise ValueError("dynamic allocation requires volatility features")

    multiplier = float(volatility["time_multiplier"])
    target = base_time_per_move * multiplier
    minimum = base_time_per_move * future_reserve_factor
    future_moves = max(0, remaining_move_capacity - 1)
    future_reserve = minimum * future_moves
    spendable_now = max(0.0, remaining_clock - future_reserve)

    # If the clock has already fallen below the desired reserve schedule, use
    # only the smaller emergency allocation.  In all cases, never allocate more
    # than the remaining total clock.
    available_now = max(minimum, spendable_now)
    return min(target, available_now, remaining_clock)
