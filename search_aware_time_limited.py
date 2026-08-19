"""Search-stability-aware time management for the V2 chess experiment.

This module is intentionally separate from ``time_limited_search.py`` so the
completed V1 experiment remains frozen and reproducible.  Fixed and V2 players
share the same evaluator, move ordering, and timed ordered Alpha-Beta core.
Only the rule for deciding whether to start another iterative-deepening level
is different.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import chess

from evaluation import evaluate_board


class SearchTimeout(Exception):
    """Raised internally when the active iteration reaches its deadline."""


STABILITY_TIME_MULTIPLIERS = {
    "stable": 0.80,
    "uncertain": 1.00,
    "unstable": 2.00,
}

# Scores use the evaluator's centipawn-like scale (one pawn = 100 points).
UNSTABLE_SCORE_CHANGE_MIN = 75
NARROW_ROOT_GAP_MAX = 25

# V2 protects its own low-instability target for every nominal future move.
FUTURE_RESERVE_FACTOR = STABILITY_TIME_MULTIPLIERS["stable"]

# A prediction must fit with this multiplicative safety allowance before the
# next full iteration is launched.  Observed growth is clamped to stop timing
# noise at very shallow depths from dominating the estimate.
NEXT_ITERATION_SAFETY_FACTOR = 1.10
MIN_ITERATION_GROWTH = 1.50
MAX_ITERATION_GROWTH = 8.00
MIN_START_MARGIN_FRACTION = 0.02
MIN_START_MARGIN_SECONDS = 0.0005


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise SearchTimeout


def _ordered_moves(board: chess.Board, deadline: float) -> List[chess.Move]:
    """Use the same deterministic ordering priorities as V1."""

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
                score += 10  # En passant has an empty destination square.

        if move.promotion is not None:
            score += 100
        if board.gives_check(move):
            score += 50

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


def _search_one_depth_detailed(
    board: chess.Board,
    depth: int,
    deadline: float,
    stats: Dict[str, int],
) -> Tuple[Optional[chess.Move], int, Dict[str, object]]:
    """Complete one depth and retain root information for V2 diagnostics.

    Alpha and beta are shared across root moves exactly as in V1.  Therefore
    non-principal root scores can be bounds rather than exact minimax scores.
    The resulting best-versus-second gap is deliberately named an *observed*
    root gap and is used only as an interpretable confidence proxy.
    """

    _check_deadline(deadline)
    moves = _ordered_moves(board, deadline)
    if not moves:
        return None, evaluate_board(board), {
            "root_move_count": 0,
            "observed_root_score_gap": None,
            "root_move_scores": [],
        }

    best_move: Optional[chess.Move] = None
    root_move_scores: List[Tuple[str, int]] = []
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

            root_move_scores.append((move.uci(), int(score)))
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        sorted_scores = sorted(
            (score for _move, score in root_move_scores), reverse=True
        )
    else:
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

            root_move_scores.append((move.uci(), int(score)))
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
        sorted_scores = sorted(score for _move, score in root_move_scores)

    observed_gap: Optional[int]
    if len(sorted_scores) >= 2:
        observed_gap = abs(int(sorted_scores[0]) - int(sorted_scores[1]))
    else:
        observed_gap = None

    return best_move, int(best_score), {
        "root_move_count": len(root_move_scores),
        "observed_root_score_gap": observed_gap,
        "root_move_scores": root_move_scores,
    }


def allocate_search_aware_targets(
    base_time_per_move: float,
    remaining_clock: float,
    remaining_move_capacity: int,
    future_reserve_factor: float = FUTURE_RESERVE_FACTOR,
) -> Dict[str, float]:
    """Return the three V2 targets while protecting future minimum budgets."""

    if base_time_per_move <= 0:
        raise ValueError("base_time_per_move must be positive")
    if remaining_clock <= 0:
        return {name: 0.0 for name in STABILITY_TIME_MULTIPLIERS}

    minimum = base_time_per_move * future_reserve_factor
    future_moves = max(0, remaining_move_capacity - 1)
    future_reserve = minimum * future_moves
    spendable_now = max(0.0, remaining_clock - future_reserve)
    available_now = max(minimum, spendable_now)

    return {
        name: min(
            base_time_per_move * multiplier,
            available_now,
            remaining_clock,
        )
        for name, multiplier in STABILITY_TIME_MULTIPLIERS.items()
    }


def classify_search_stability(
    iteration_records: List[Dict[str, object]],
) -> Dict[str, object]:
    """Classify the latest completed iteration as stable/uncertain/unstable.

    The rule is frozen before the V2 formal run:

    Three binary instability signals are counted: the best move changed, the
    score changed by at least 75, and the observed best-versus-second root gap
    is at most 25.  Zero or one signal is ``stable`` (0.8x), two signals are
    ``uncertain`` (1.0x), and all three signals are ``unstable`` (2.0x).

    Requiring corroboration prevents one noisy Alpha-Beta root bound from
    independently forcing the maximum allocation.  The labels therefore mean
    low, medium, and high *evidence of search instability*, respectively.

    One forced legal move is stable.  With only one completed depth there is no
    cross-depth evidence, so the position remains uncertain.
    """

    if not iteration_records:
        return {
            "stability_class": "uncertain",
            "time_multiplier": STABILITY_TIME_MULTIPLIERS["uncertain"],
            "best_move_changed": None,
            "absolute_score_change": None,
            "observed_root_score_gap": None,
            "instability_signal_count": 0,
            "reason": "no_completed_iteration",
        }

    current = iteration_records[-1]
    current_move = str(current["best_move_uci"])
    current_score = int(current["score"])
    root_count = int(current["root_move_count"])
    gap_value = current["observed_root_score_gap"]
    gap = None if gap_value is None else int(gap_value)

    if root_count <= 1:
        stability_class = "stable"
        return {
            "stability_class": stability_class,
            "time_multiplier": STABILITY_TIME_MULTIPLIERS[stability_class],
            "best_move_changed": False,
            "absolute_score_change": 0 if len(iteration_records) >= 2 else None,
            "observed_root_score_gap": gap,
            "instability_signal_count": 0,
            "reason": "forced_move",
        }

    if len(iteration_records) < 2:
        stability_class = "uncertain"
        reason = "insufficient_cross_depth_history"
        return {
            "stability_class": stability_class,
            "time_multiplier": STABILITY_TIME_MULTIPLIERS[stability_class],
            "best_move_changed": None,
            "absolute_score_change": None,
            "observed_root_score_gap": gap,
            "instability_signal_count": 0,
            "reason": reason,
        }

    previous = iteration_records[-2]
    previous_move = str(previous["best_move_uci"])
    previous_score = int(previous["score"])
    move_changed = current_move != previous_move
    score_change = abs(current_score - previous_score)

    instability_reasons = []
    if move_changed:
        instability_reasons.append("best_move_changed")
    if score_change >= UNSTABLE_SCORE_CHANGE_MIN:
        instability_reasons.append("large_score_change")
    if gap is not None and gap <= NARROW_ROOT_GAP_MAX:
        instability_reasons.append("narrow_root_gap")

    signal_count = len(instability_reasons)
    if signal_count == 3:
        stability_class = "unstable"
    elif signal_count == 2:
        stability_class = "uncertain"
    else:
        stability_class = "stable"
    reason = (
        "+".join(instability_reasons)
        if instability_reasons
        else "no_instability_signal"
    )

    return {
        "stability_class": stability_class,
        "time_multiplier": STABILITY_TIME_MULTIPLIERS[stability_class],
        "best_move_changed": move_changed,
        "absolute_score_change": score_change,
        "observed_root_score_gap": gap,
        "instability_signal_count": signal_count,
        "reason": reason,
    }


def estimate_next_iteration_time(
    iteration_records: List[Dict[str, object]],
    safety_factor: float = NEXT_ITERATION_SAFETY_FACTOR,
) -> Optional[Dict[str, float]]:
    """Estimate the next full iteration from observed time and node growth."""

    if len(iteration_records) < 2:
        return None

    previous = iteration_records[-2]
    current = iteration_records[-1]
    previous_time = max(float(previous["iteration_time"]), 1e-9)
    current_time = max(float(current["iteration_time"]), 1e-9)
    previous_nodes = max(int(previous["iteration_nodes"]), 1)
    current_nodes = max(int(current["iteration_nodes"]), 1)

    time_growth = current_time / previous_time
    node_growth = current_nodes / previous_nodes
    # Time and node ratios are two noisy observations of the same expansion.
    # Their geometric mean is less dominated by shallow timing overhead than
    # taking the larger value, while the clamp remains conservative.
    raw_growth = math.sqrt(time_growth * node_growth)
    clamped_growth = min(
        MAX_ITERATION_GROWTH,
        max(MIN_ITERATION_GROWTH, raw_growth),
    )
    predicted_time = current_time * clamped_growth * safety_factor

    return {
        "predicted_seconds": predicted_time,
        "time_growth": time_growth,
        "node_growth": node_growth,
        "clamped_growth": clamped_growth,
        "safety_factor": safety_factor,
    }


def _empty_stats(stop_reason: str) -> Dict[str, object]:
    return {
        "nodes": 0,
        "cutoffs": 0,
        "completed_depth": 0,
        "attempted_depth": 0,
        "timed_out": False,
        "completed_iterations": 0,
        "iteration_records": [],
        "iteration_nodes": [],
        "iteration_times": [],
        "iteration_best_moves": [],
        "iteration_scores": [],
        "iteration_root_gaps": [],
        "incomplete_iteration_nodes": 0,
        "incomplete_node_fraction": 0.0,
        "stability_class": "uncertain",
        "time_multiplier": STABILITY_TIME_MULTIPLIERS["uncertain"],
        "best_move_changed": None,
        "absolute_score_change": None,
        "observed_root_score_gap": None,
        "instability_signal_count": 0,
        "stability_reason": "no_completed_iteration",
        "time_targets": {name: 0.0 for name in STABILITY_TIME_MULTIPLIERS},
        "committed_time_limit": 0.0,
        "hard_time_cap": 0.0,
        "final_policy_target": 0.0,
        "predicted_next_iteration_time": None,
        "prediction_time_growth": None,
        "prediction_node_growth": None,
        "prediction_clamped_growth": None,
        "policy_stopped_at_iteration_boundary": False,
        "stop_reason": stop_reason,
    }


def find_best_move_time_managed(
    board: chess.Board,
    strategy: str,
    base_time_per_move: float,
    remaining_clock: float,
    remaining_move_capacity: int,
    max_depth: int = 8,
    decision_start: Optional[float] = None,
) -> Tuple[Optional[chess.Move], Optional[int], Dict[str, object]]:
    """Run fixed or V2 search using the same timed Alpha-Beta core.

    ``fixed`` starts deeper iterations until its nominal deadline interrupts
    one. ``search_aware`` begins with the 1.0x target, reclassifies after each
    completed depth, and starts another depth only when the predicted full
    iteration fits inside the selected 0.8x/1.0x/2.0x target.
    """

    if strategy not in {"fixed", "search_aware"}:
        raise ValueError("strategy must be 'fixed' or 'search_aware'")
    if base_time_per_move <= 0:
        raise ValueError("base_time_per_move must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, evaluate_board(board), _empty_stats("terminal_position")

    best_move = legal_moves[0]
    best_score: Optional[int] = None
    if remaining_clock <= 0:
        stats = _empty_stats("clock_exhausted_depth_zero_fallback")
        stats["timed_out"] = True
        return best_move, best_score, stats

    start = time.perf_counter() if decision_start is None else decision_start
    fixed_limit = min(base_time_per_move, remaining_clock)
    time_targets = allocate_search_aware_targets(
        base_time_per_move,
        remaining_clock,
        remaining_move_capacity,
    )

    total_stats: Dict[str, int] = {"nodes": 0, "cutoffs": 0}
    iteration_records: List[Dict[str, object]] = []
    completed_depth = 0
    attempted_depth = 0
    timed_out = False
    policy_stopped = False
    stop_reason = "max_depth_completed"
    committed_time_limit = 0.0
    prediction_at_stop: Optional[Dict[str, float]] = None
    final_stability = classify_search_stability(iteration_records)

    for depth in range(1, max_depth + 1):
        if strategy == "fixed":
            current_target = fixed_limit
            prediction = None
        else:
            final_stability = classify_search_stability(iteration_records)
            current_target = time_targets[str(final_stability["stability_class"])]
            prediction = estimate_next_iteration_time(iteration_records)

            if depth > 1:
                elapsed = time.perf_counter() - start
                remaining_target = current_target - elapsed
                start_margin = max(
                    MIN_START_MARGIN_SECONDS,
                    base_time_per_move * MIN_START_MARGIN_FRACTION,
                )
                if remaining_target <= start_margin:
                    policy_stopped = True
                    stop_reason = "selected_time_target_reached"
                    prediction_at_stop = prediction
                    break
                if (
                    prediction is not None
                    and prediction["predicted_seconds"] > remaining_target
                ):
                    policy_stopped = True
                    stop_reason = "predicted_next_iteration_exceeds_target"
                    prediction_at_stop = prediction
                    break

        attempted_depth = depth
        committed_time_limit = max(committed_time_limit, current_target)
        deadline = start + current_target
        nodes_before = total_stats["nodes"]
        iteration_start = time.perf_counter()
        try:
            move, score, details = _search_one_depth_detailed(
                board,
                depth,
                deadline,
                total_stats,
            )
        except SearchTimeout:
            timed_out = True
            stop_reason = "deadline_timeout"
            prediction_at_stop = prediction
            break

        iteration_time = time.perf_counter() - iteration_start
        iteration_nodes = total_stats["nodes"] - nodes_before
        if move is not None:
            best_move = move
            best_score = score
        completed_depth = depth
        iteration_records.append(
            {
                "depth": depth,
                "best_move_uci": "" if move is None else move.uci(),
                "score": score,
                "iteration_time": iteration_time,
                "iteration_nodes": iteration_nodes,
                "root_move_count": details["root_move_count"],
                "observed_root_score_gap": details[
                    "observed_root_score_gap"
                ],
                "root_move_scores": details["root_move_scores"],
                "target_time_before_iteration": current_target,
                "predicted_time_before_iteration": (
                    None if prediction is None else prediction["predicted_seconds"]
                ),
            }
        )
        final_stability = classify_search_stability(iteration_records)

    completed_nodes = sum(
        int(record["iteration_nodes"]) for record in iteration_records
    )
    incomplete_nodes = max(0, total_stats["nodes"] - completed_nodes)
    incomplete_fraction = (
        incomplete_nodes / total_stats["nodes"] if total_stats["nodes"] else 0.0
    )
    final_stability = classify_search_stability(iteration_records)

    if strategy == "fixed":
        final_policy_target = fixed_limit
        hard_time_cap = fixed_limit
        final_multiplier = 1.0
    else:
        final_class = str(final_stability["stability_class"])
        final_policy_target = time_targets[final_class]
        hard_time_cap = time_targets["unstable"]
        final_multiplier = STABILITY_TIME_MULTIPLIERS[final_class]

    if completed_depth == max_depth and not timed_out and not policy_stopped:
        stop_reason = "max_depth_completed"

    prediction_values = prediction_at_stop
    if prediction_values is None:
        prediction_values = estimate_next_iteration_time(iteration_records)

    return best_move, best_score, {
        "nodes": total_stats["nodes"],
        "cutoffs": total_stats["cutoffs"],
        "completed_depth": completed_depth,
        "attempted_depth": attempted_depth,
        "timed_out": timed_out,
        "completed_iterations": len(iteration_records),
        "iteration_records": iteration_records,
        "iteration_nodes": [
            int(record["iteration_nodes"]) for record in iteration_records
        ],
        "iteration_times": [
            float(record["iteration_time"]) for record in iteration_records
        ],
        "iteration_best_moves": [
            str(record["best_move_uci"]) for record in iteration_records
        ],
        "iteration_scores": [
            int(record["score"]) for record in iteration_records
        ],
        "iteration_root_gaps": [
            record["observed_root_score_gap"] for record in iteration_records
        ],
        "incomplete_iteration_nodes": incomplete_nodes,
        "incomplete_node_fraction": incomplete_fraction,
        "stability_class": final_stability["stability_class"],
        "time_multiplier": final_multiplier,
        "best_move_changed": final_stability["best_move_changed"],
        "absolute_score_change": final_stability["absolute_score_change"],
        "observed_root_score_gap": final_stability[
            "observed_root_score_gap"
        ],
        "instability_signal_count": final_stability[
            "instability_signal_count"
        ],
        "stability_reason": final_stability["reason"],
        "time_targets": time_targets,
        "committed_time_limit": committed_time_limit,
        "hard_time_cap": hard_time_cap,
        "final_policy_target": final_policy_target,
        "predicted_next_iteration_time": (
            None
            if prediction_values is None
            else prediction_values["predicted_seconds"]
        ),
        "prediction_time_growth": (
            None if prediction_values is None else prediction_values["time_growth"]
        ),
        "prediction_node_growth": (
            None if prediction_values is None else prediction_values["node_growth"]
        ),
        "prediction_clamped_growth": (
            None
            if prediction_values is None
            else prediction_values["clamped_growth"]
        ),
        "policy_stopped_at_iteration_boundary": policy_stopped,
        "stop_reason": stop_reason,
    }
