"""Formal Fixed-versus-V2 search-aware time-allocation experiment.

Default formal design:
    2 nominal budgets x 12 openings x 2 colours = 48 games.

The completed V1 files are not imported or modified as results.  This runner
reuses only their validated opening definitions and generic CSV helpers.  Both
players receive equal initial clocks and use the same evaluator, deterministic
move ordering, and timed ordered Alpha-Beta core.  The only experimental
difference is the iterative-deepening time-management policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import chess

from evaluation import evaluate_board
from search_aware_time_limited import (
    FUTURE_RESERVE_FACTOR,
    MAX_ITERATION_GROWTH,
    MIN_ITERATION_GROWTH,
    NARROW_ROOT_GAP_MAX,
    NEXT_ITERATION_SAFETY_FACTOR,
    STABILITY_TIME_MULTIPLIERS,
    UNSTABLE_SCORE_CHANGE_MIN,
    find_best_move_time_managed,
)


FORMAL_BASE_BUDGETS = [0.10, 0.20]


# Frozen copy of the 12 opening lines used in the earlier experiments.  Keeping
# them here makes the V2 package runnable without importing or changing V1.
OPENING_LINES = [
    {"name": "starting_position", "family": "starting_position", "moves": []},
    {"name": "open_game", "family": "open_game", "moves": ["e4", "e5"]},
    {
        "name": "italian_game",
        "family": "open_game",
        "moves": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"],
    },
    {
        "name": "ruy_lopez",
        "family": "open_game",
        "moves": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"],
    },
    {
        "name": "scotch_game",
        "family": "open_game",
        "moves": ["e4", "e5", "Nf3", "Nc6", "d4", "exd4"],
    },
    {
        "name": "sicilian_defence",
        "family": "semi_open_game",
        "moves": ["e4", "c5", "Nf3", "d6", "d4", "cxd4"],
    },
    {
        "name": "french_defence",
        "family": "semi_open_game",
        "moves": ["e4", "e6", "d4", "d5", "Nc3", "Nf6"],
    },
    {
        "name": "caro_kann_defence",
        "family": "semi_open_game",
        "moves": ["e4", "c6", "d4", "d5", "Nc3", "dxe4"],
    },
    {
        "name": "queens_gambit_declined",
        "family": "closed_game",
        "moves": ["d4", "d5", "c4", "e6", "Nc3", "Nf6"],
    },
    {
        "name": "slav_defence",
        "family": "closed_game",
        "moves": ["d4", "d5", "c4", "c6", "Nf3", "Nf6"],
    },
    {
        "name": "kings_indian_defence",
        "family": "indian_game",
        "moves": ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7"],
    },
    {
        "name": "english_opening",
        "family": "flank_opening",
        "moves": ["c4", "e5", "Nc3", "Nf6", "g3", "d5"],
    },
]


def apply_opening(moves: Iterable[str]) -> chess.Board:
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board


def validate_opening_lines(openings: List[Dict[str, object]]) -> None:
    names = set()
    for opening in openings:
        name = str(opening["name"])
        moves = list(opening["moves"])
        if name in names:
            raise ValueError(f"Duplicate opening name: {name}")
        names.add(name)
        if len(moves) % 2:
            raise ValueError(f"{name} has an odd number of opening plies")
        board = apply_opening(moves)
        if board.turn != chess.WHITE:
            raise ValueError(f"{name} does not leave White to move")
        if board.outcome(claim_draw=True) is not None:
            raise ValueError(f"{name} is already terminal")


def safe_average(total: float, count: int) -> float:
    return total / count if count else 0.0


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_outcome(board: chess.Board) -> Tuple[Optional[str], Optional[str]]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None, None
    return outcome.result(), outcome.termination.name.lower()


def _csv_value(value: object) -> object:
    return "" if value is None else value


def score_for_v2(result: str, v2_colour: str) -> Optional[float]:
    if result == "*":
        return None
    if result == "1/2-1/2":
        return 0.5
    if v2_colour == "white":
        return 1.0 if result == "1-0" else 0.0
    return 1.0 if result == "0-1" else 0.0


def _new_side_totals() -> Dict[str, object]:
    return {
        "moves": 0,
        "nodes": 0,
        "cutoffs": 0,
        "incomplete_iteration_nodes": 0,
        "committed_time": 0.0,
        "decision_time": 0.0,
        "completed_depth_sum": 0,
        "maximum_completed_depth": 0,
        "depth_zero_fallbacks": 0,
        "deadline_overruns": 0,
        "deadline_timeouts": 0,
        "policy_boundary_stops": 0,
        "stable_positions": 0,
        "uncertain_positions": 0,
        "unstable_positions": 0,
        "best_move_changes": 0,
    }


def play_game(
    opening: Dict[str, object],
    base_time_per_move: float,
    v2_colour: str,
    experiment_timestamp: str,
    game_id: str,
    max_plies: int = 160,
    max_depth: int = 8,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    board = apply_opening(opening["moves"])
    opening_fen = board.fen()
    player_move_capacity = math.ceil(max_plies / 2)
    initial_clock = base_time_per_move * player_move_capacity
    strategies = {
        v2_colour: "search_aware",
        "black" if v2_colour == "white" else "white": "fixed",
    }
    clocks = {"white": initial_clock, "black": initial_clock}
    totals = {"white": _new_side_totals(), "black": _new_side_totals()}
    move_rows: List[Dict[str, object]] = []
    engine_uci: List[str] = []
    engine_san: List[str] = []
    engine_plies = 0
    game_wall_start = time.perf_counter()

    while board.outcome(claim_draw=True) is None and engine_plies < max_plies:
        side = "white" if board.turn == chess.WHITE else "black"
        strategy = strategies[side]
        side_totals = totals[side]
        clock_before = clocks[side]
        remaining_move_capacity = max(
            1,
            player_move_capacity - int(side_totals["moves"]),
        )

        decision_start = time.perf_counter()
        move, score, stats = find_best_move_time_managed(
            board,
            strategy=strategy,
            base_time_per_move=base_time_per_move,
            remaining_clock=clock_before,
            remaining_move_capacity=remaining_move_capacity,
            max_depth=max_depth,
            decision_start=decision_start,
        )
        decision_time = time.perf_counter() - decision_start

        if move is None:
            raise RuntimeError(f"No move in non-terminal game {game_id}")
        if move not in board.legal_moves:
            raise RuntimeError(
                f"Illegal move {move} in game {game_id}; FEN={board.fen()}"
            )

        san = board.san(move)
        clock_after = max(0.0, clock_before - decision_time)
        clocks[side] = clock_after
        committed_time = float(stats["committed_time_limit"])
        overrun = max(0.0, decision_time - committed_time)
        completed_depth = int(stats["completed_depth"])
        stability_class = str(stats["stability_class"])

        side_totals["moves"] = int(side_totals["moves"]) + 1
        side_totals["nodes"] = int(side_totals["nodes"]) + int(stats["nodes"])
        side_totals["cutoffs"] = int(side_totals["cutoffs"]) + int(
            stats["cutoffs"]
        )
        side_totals["incomplete_iteration_nodes"] = int(
            side_totals["incomplete_iteration_nodes"]
        ) + int(stats["incomplete_iteration_nodes"])
        side_totals["committed_time"] = float(
            side_totals["committed_time"]
        ) + committed_time
        side_totals["decision_time"] = float(
            side_totals["decision_time"]
        ) + decision_time
        side_totals["completed_depth_sum"] = int(
            side_totals["completed_depth_sum"]
        ) + completed_depth
        side_totals["maximum_completed_depth"] = max(
            int(side_totals["maximum_completed_depth"]),
            completed_depth,
        )
        side_totals["depth_zero_fallbacks"] = int(
            side_totals["depth_zero_fallbacks"]
        ) + int(completed_depth == 0)
        side_totals["deadline_overruns"] = int(
            side_totals["deadline_overruns"]
        ) + int(overrun > 0)
        side_totals["deadline_timeouts"] = int(
            side_totals["deadline_timeouts"]
        ) + int(bool(stats["timed_out"]))
        side_totals["policy_boundary_stops"] = int(
            side_totals["policy_boundary_stops"]
        ) + int(bool(stats["policy_stopped_at_iteration_boundary"]))
        side_totals[f"{stability_class}_positions"] = int(
            side_totals[f"{stability_class}_positions"]
        ) + 1
        side_totals["best_move_changes"] = int(
            side_totals["best_move_changes"]
        ) + int(stats["best_move_changed"] is True)

        targets = dict(stats["time_targets"])
        move_rows.append(
            {
                "experiment_timestamp": experiment_timestamp,
                "game_id": game_id,
                "opening_name": opening["name"],
                "base_time_per_move": base_time_per_move,
                "engine_ply_index": engine_plies + 1,
                "total_ply_from_start": len(opening["moves"]) + engine_plies + 1,
                "side": side,
                "strategy": strategy,
                "v2_colour": v2_colour,
                "fen_before_move": board.fen(),
                "move_uci": move.uci(),
                "move_san": san,
                "score_white_perspective": _csv_value(score),
                "initial_clock": initial_clock,
                "clock_before": clock_before,
                "remaining_move_capacity": remaining_move_capacity,
                "stable_time_target": targets["stable"],
                "uncertain_time_target": targets["uncertain"],
                "unstable_time_target": targets["unstable"],
                "final_policy_target": stats["final_policy_target"],
                "committed_time_limit": committed_time,
                "hard_time_cap": stats["hard_time_cap"],
                "decision_time_charged": decision_time,
                "deadline_overrun": overrun,
                "clock_after": clock_after,
                "completed_depth": completed_depth,
                "attempted_depth": stats["attempted_depth"],
                "timed_out": stats["timed_out"],
                "stop_reason": stats["stop_reason"],
                "policy_stopped_at_iteration_boundary": stats[
                    "policy_stopped_at_iteration_boundary"
                ],
                "nodes": stats["nodes"],
                "cutoffs": stats["cutoffs"],
                "incomplete_iteration_nodes": stats[
                    "incomplete_iteration_nodes"
                ],
                "incomplete_node_fraction": stats[
                    "incomplete_node_fraction"
                ],
                "iteration_nodes": "|".join(
                    str(value) for value in stats["iteration_nodes"]
                ),
                "iteration_times": "|".join(
                    f"{float(value):.9f}" for value in stats["iteration_times"]
                ),
                "iteration_best_moves": "|".join(
                    str(value) for value in stats["iteration_best_moves"]
                ),
                "iteration_scores": "|".join(
                    str(value) for value in stats["iteration_scores"]
                ),
                "iteration_root_gaps": "|".join(
                    "" if value is None else str(value)
                    for value in stats["iteration_root_gaps"]
                ),
                "stability_class": stability_class,
                "time_multiplier": (
                    stats["time_multiplier"]
                    if strategy == "search_aware"
                    else 1.0
                ),
                "best_move_changed": _csv_value(stats["best_move_changed"]),
                "absolute_score_change": _csv_value(
                    stats["absolute_score_change"]
                ),
                "observed_root_score_gap": _csv_value(
                    stats["observed_root_score_gap"]
                ),
                "instability_signal_count": stats[
                    "instability_signal_count"
                ],
                "stability_reason": stats["stability_reason"],
                "predicted_next_iteration_time": _csv_value(
                    stats["predicted_next_iteration_time"]
                ),
                "prediction_time_growth": _csv_value(
                    stats["prediction_time_growth"]
                ),
                "prediction_node_growth": _csv_value(
                    stats["prediction_node_growth"]
                ),
                "prediction_clamped_growth": _csv_value(
                    stats["prediction_clamped_growth"]
                ),
            }
        )

        engine_uci.append(move.uci())
        engine_san.append(san)
        board.push(move)
        engine_plies += 1

    game_wall_time = time.perf_counter() - game_wall_start
    result, termination = get_outcome(board)
    if result is None:
        completed = False
        result = "*"
        termination = "max_plies_truncation"
        final_evaluation = evaluate_board(board)
    else:
        completed = True
        outcome = board.outcome(claim_draw=True)
        final_evaluation = (
            0
            if outcome is not None and outcome.winner is None
            else evaluate_board(board)
        )

    v2_score = score_for_v2(result, v2_colour)
    fixed_colour = "black" if v2_colour == "white" else "white"
    v2_totals = totals[v2_colour]
    fixed_totals = totals[fixed_colour]

    def role_fields(role: str, data: Dict[str, object], colour: str):
        moves = int(data["moves"])
        nodes = int(data["nodes"])
        incomplete_nodes = int(data["incomplete_iteration_nodes"])
        return {
            f"{role}_colour": colour,
            f"{role}_move_count": moves,
            f"{role}_total_nodes": nodes,
            f"{role}_average_nodes_per_move": safe_average(nodes, moves),
            f"{role}_total_cutoffs": data["cutoffs"],
            f"{role}_incomplete_iteration_nodes": incomplete_nodes,
            f"{role}_incomplete_node_fraction": safe_average(
                incomplete_nodes,
                nodes,
            ),
            f"{role}_total_committed_time": data["committed_time"],
            f"{role}_total_decision_time": data["decision_time"],
            f"{role}_average_decision_time_per_move": safe_average(
                float(data["decision_time"]),
                moves,
            ),
            f"{role}_average_completed_depth": safe_average(
                float(data["completed_depth_sum"]),
                moves,
            ),
            f"{role}_maximum_completed_depth": data["maximum_completed_depth"],
            f"{role}_depth_zero_fallbacks": data["depth_zero_fallbacks"],
            f"{role}_deadline_overruns": data["deadline_overruns"],
            f"{role}_deadline_timeouts": data["deadline_timeouts"],
            f"{role}_policy_boundary_stops": data["policy_boundary_stops"],
            f"{role}_stable_positions": data["stable_positions"],
            f"{role}_uncertain_positions": data["uncertain_positions"],
            f"{role}_unstable_positions": data["unstable_positions"],
            f"{role}_best_move_changes": data["best_move_changes"],
            f"{role}_clock_used": initial_clock - clocks[colour],
            f"{role}_clock_remaining": clocks[colour],
        }

    game_row = {
        "experiment_timestamp": experiment_timestamp,
        "game_id": game_id,
        "opening_name": opening["name"],
        "opening_family": opening["family"],
        "opening_moves_san": " ".join(opening["moves"]),
        "opening_fen": opening_fen,
        "base_time_per_move": base_time_per_move,
        "initial_clock_per_player": initial_clock,
        "completed_game": completed,
        "result": result,
        "termination_reason": termination,
        "v2_score": _csv_value(v2_score),
        "engine_plies_played": engine_plies,
        "total_plies_from_start": len(opening["moves"]) + engine_plies,
        "engine_moves_uci": " ".join(engine_uci),
        "engine_moves_san": " ".join(engine_san),
        "full_game_san": " ".join(list(opening["moves"]) + engine_san),
        "final_fen": board.fen(),
        "final_evaluation_white_perspective": final_evaluation,
        "game_wall_time": game_wall_time,
        **role_fields("v2", v2_totals, v2_colour),
        **role_fields("fixed", fixed_totals, fixed_colour),
    }
    return game_row, move_rows


def summarise(game_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[float, Dict[str, object]] = {}
    for row in game_rows:
        budget = float(row["base_time_per_move"])
        data = grouped.setdefault(
            budget,
            {
                "base_time_per_move": budget,
                "scheduled_games": 0,
                "completed_games": 0,
                "truncated_games": 0,
                "v2_wins": 0,
                "draws": 0,
                "v2_losses": 0,
                "completed_score": 0.0,
                "v2_white_completed_games": 0,
                "v2_white_score": 0.0,
                "v2_black_completed_games": 0,
                "v2_black_score": 0.0,
                "v2_moves": 0,
                "fixed_moves": 0,
                "v2_nodes": 0,
                "fixed_nodes": 0,
                "v2_incomplete_iteration_nodes": 0,
                "fixed_incomplete_iteration_nodes": 0,
                "v2_decision_time": 0.0,
                "fixed_decision_time": 0.0,
                "v2_completed_depth_weighted_sum": 0.0,
                "fixed_completed_depth_weighted_sum": 0.0,
                "v2_clock_used": 0.0,
                "fixed_clock_used": 0.0,
                "v2_depth_zero_fallbacks": 0,
                "fixed_depth_zero_fallbacks": 0,
                "v2_deadline_timeouts": 0,
                "fixed_deadline_timeouts": 0,
                "v2_policy_boundary_stops": 0,
                "fixed_policy_boundary_stops": 0,
                "v2_stable_positions": 0,
                "v2_uncertain_positions": 0,
                "v2_unstable_positions": 0,
            },
        )

        data["scheduled_games"] += 1
        for role in ("v2", "fixed"):
            moves = int(row[f"{role}_move_count"])
            data[f"{role}_moves"] += moves
            data[f"{role}_nodes"] += int(row[f"{role}_total_nodes"])
            data[f"{role}_incomplete_iteration_nodes"] += int(
                row[f"{role}_incomplete_iteration_nodes"]
            )
            data[f"{role}_decision_time"] += float(
                row[f"{role}_total_decision_time"]
            )
            data[f"{role}_completed_depth_weighted_sum"] += (
                float(row[f"{role}_average_completed_depth"]) * moves
            )
            data[f"{role}_clock_used"] += float(row[f"{role}_clock_used"])
            data[f"{role}_depth_zero_fallbacks"] += int(
                row[f"{role}_depth_zero_fallbacks"]
            )
            data[f"{role}_deadline_timeouts"] += int(
                row[f"{role}_deadline_timeouts"]
            )
            data[f"{role}_policy_boundary_stops"] += int(
                row[f"{role}_policy_boundary_stops"]
            )
            if role == "v2":
                for stability_class in ("stable", "uncertain", "unstable"):
                    data[f"v2_{stability_class}_positions"] += int(
                        row[f"v2_{stability_class}_positions"]
                    )

        if not row["completed_game"]:
            data["truncated_games"] += 1
            continue

        data["completed_games"] += 1
        score = float(row["v2_score"])
        data["completed_score"] += score
        if score == 1.0:
            data["v2_wins"] += 1
        elif score == 0.5:
            data["draws"] += 1
        else:
            data["v2_losses"] += 1

        colour = str(row["v2_colour"])
        data[f"v2_{colour}_completed_games"] += 1
        data[f"v2_{colour}_score"] += score

    summaries = []
    for budget in sorted(grouped):
        data = grouped[budget]
        summaries.append(
            {
                **data,
                "completed_score_rate": safe_average(
                    float(data["completed_score"]),
                    int(data["completed_games"]),
                ),
                "score_rate_if_truncations_count_as_half": safe_average(
                    float(data["completed_score"])
                    + 0.5 * int(data["truncated_games"]),
                    int(data["scheduled_games"]),
                ),
                "v2_white_score_rate": safe_average(
                    float(data["v2_white_score"]),
                    int(data["v2_white_completed_games"]),
                ),
                "v2_black_score_rate": safe_average(
                    float(data["v2_black_score"]),
                    int(data["v2_black_completed_games"]),
                ),
                "v2_average_nodes_per_move": safe_average(
                    float(data["v2_nodes"]),
                    int(data["v2_moves"]),
                ),
                "fixed_average_nodes_per_move": safe_average(
                    float(data["fixed_nodes"]),
                    int(data["fixed_moves"]),
                ),
                "v2_incomplete_node_fraction": safe_average(
                    float(data["v2_incomplete_iteration_nodes"]),
                    int(data["v2_nodes"]),
                ),
                "fixed_incomplete_node_fraction": safe_average(
                    float(data["fixed_incomplete_iteration_nodes"]),
                    int(data["fixed_nodes"]),
                ),
                "v2_average_decision_time_per_move": safe_average(
                    float(data["v2_decision_time"]),
                    int(data["v2_moves"]),
                ),
                "fixed_average_decision_time_per_move": safe_average(
                    float(data["fixed_decision_time"]),
                    int(data["fixed_moves"]),
                ),
                "v2_average_completed_depth": safe_average(
                    float(data["v2_completed_depth_weighted_sum"]),
                    int(data["v2_moves"]),
                ),
                "fixed_average_completed_depth": safe_average(
                    float(data["fixed_completed_depth_weighted_sum"]),
                    int(data["fixed_moves"]),
                ),
            }
        )
    return summaries


def run_search_aware_experiment(
    base_budgets: Optional[List[float]] = None,
    max_plies: int = 160,
    max_depth: int = 8,
    opening_limit: Optional[int] = None,
    output_dir: str = "results/search_aware_v2",
    experiment_name: str = "formal_search_aware_time_allocation",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, str]]:
    if base_budgets is None:
        base_budgets = list(FORMAL_BASE_BUDGETS)
    if any(value <= 0 for value in base_budgets):
        raise ValueError("All base budgets must be positive")

    openings = OPENING_LINES[:opening_limit] if opening_limit else OPENING_LINES
    validate_opening_lines(openings)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"search_aware_v2_{timestamp}"
    paths = {
        "game_checkpoint": os.path.join(output_dir, f"{stem}_game_checkpoint.csv"),
        "move_checkpoint": os.path.join(output_dir, f"{stem}_move_checkpoint.csv"),
        "game_results": os.path.join(output_dir, f"{stem}_game_results.csv"),
        "move_results": os.path.join(output_dir, f"{stem}_move_results.csv"),
        "summary": os.path.join(output_dir, f"{stem}_summary.csv"),
        "metadata": os.path.join(output_dir, f"{stem}_metadata.json"),
    }

    games: List[Dict[str, object]] = []
    moves: List[Dict[str, object]] = []
    total_games = len(base_budgets) * len(openings) * 2
    game_number = 0
    experiment_start = time.perf_counter()
    print(f"Validated {len(openings)} openings; scheduled games={total_games}.")

    for budget in base_budgets:
        for opening in openings:
            for v2_colour in ("white", "black"):
                game_number += 1
                budget_ms = int(round(budget * 1000))
                game_id = f"sa{budget_ms}ms_{opening['name']}_v2_{v2_colour}"
                print(f"\n[{game_number}/{total_games}] {game_id}")
                game, game_moves = play_game(
                    opening=opening,
                    base_time_per_move=budget,
                    v2_colour=v2_colour,
                    experiment_timestamp=timestamp,
                    game_id=game_id,
                    max_plies=max_plies,
                    max_depth=max_depth,
                )
                games.append(game)
                moves.extend(game_moves)
                save_csv(paths["game_checkpoint"], games)
                save_csv(paths["move_checkpoint"], moves)

                score = game["v2_score"] if game["v2_score"] != "" else "excluded"
                print(
                    f"result={game['result']}; termination={game['termination_reason']}; "
                    f"V2 score={score}; plies={game['engine_plies_played']}; "
                    f"wall={game['game_wall_time']:.2f}s"
                )

    summaries = summarise(games)
    save_csv(paths["game_results"], games)
    save_csv(paths["move_results"], moves)
    save_csv(paths["summary"], summaries)

    metadata = {
        "experiment_timestamp": timestamp,
        "experiment_name": experiment_name,
        "design_status": "V2 rules frozen before the formal 48-game run",
        "research_comparison": "fixed allocation versus V2 search-stability-aware allocation",
        "base_time_per_move_seconds": base_budgets,
        "opening_count": len(openings),
        "games_per_budget": len(openings) * 2,
        "scheduled_games": total_games,
        "max_engine_plies": max_plies,
        "max_iterative_depth": max_depth,
        "player_move_capacity": math.ceil(max_plies / 2),
        "initial_clock_formula": "base_time_per_move * ceil(max_engine_plies / 2)",
        "fairness_control": (
            "Both players receive the same initial total clock and use the same "
            "evaluator, deterministic move ordering, timed ordered Alpha-Beta "
            "core, maximum depth, openings, and colour-pairing rules. Only the "
            "iterative-deepening time controller differs."
        ),
        "fixed_policy": (
            "Use min(base_time_per_move, remaining_clock) and start successively "
            "deeper iterations until the deadline interrupts one."
        ),
        "v2_policy": (
            "Begin with the uncertain 1.0x target; after completed iterations, "
            "classify search stability and use a 0.8x, 1.0x, or 2.0x target. "
            "Start the next depth only when its predicted full cost fits."
        ),
        "stability_time_multipliers": STABILITY_TIME_MULTIPLIERS,
        "future_reserve_factor": FUTURE_RESERVE_FACTOR,
        "stability_thresholds_centipawn_like_points": {
            "unstable_score_change_min": UNSTABLE_SCORE_CHANGE_MIN,
            "narrow_observed_root_gap_max": NARROW_ROOT_GAP_MAX,
        },
        "stability_rule": {
            "unstable": (
                "all three instability signals are present: best move changed, "
                "absolute score change >= 75, and observed root score gap <= 25"
            ),
            "uncertain": (
                "exactly two of the three instability signals are present, or "
                "cross-depth history is not yet available"
            ),
            "stable": "zero or one of the three instability signals is present",
            "forced_move": "stable",
            "interpretation": (
                "stable, uncertain, and unstable mean low, medium, and high "
                "corroborated evidence of search instability"
            ),
        },
        "observed_root_gap_caveat": (
            "Root alpha/beta bounds are shared exactly as in V1, so non-principal "
            "root scores can be bounds. The gap is a confidence proxy, not an "
            "exact minimax margin; it cannot trigger the 2.0x allocation alone."
        ),
        "next_iteration_prediction": {
            "formula": (
                "last_iteration_time * geometric_mean(time_growth, node_growth) "
                "* safety_factor, with growth clamped"
            ),
            "safety_factor": NEXT_ITERATION_SAFETY_FACTOR,
            "minimum_growth": MIN_ITERATION_GROWTH,
            "maximum_growth": MAX_ITERATION_GROWTH,
        },
        "formal_run_rule": (
            "Do not change budgets, thresholds, multipliers, prediction constants, "
            "openings, maximum depth, or maximum plies after the formal run begins; "
            "do not tune V2 after observing formal outcomes."
        ),
        "truncation_policy": (
            'Games reaching max_engine_plies use result="*" and are excluded '
            "from the primary completed-game score rate."
        ),
        "primary_strength_metric": "V2 score rate over completed games.",
        "primary_efficiency_metric": (
            "Incomplete-iteration node fraction compared with Fixed."
        ),
        "secondary_metrics": [
            "average completed depth",
            "actual decision time deducted from equal clocks",
            "nodes per move",
            "policy boundary-stop rate",
            "stability-class distribution",
            "depth-0 fallbacks",
        ],
        "timer": "time.perf_counter",
        "paths": paths,
        "python_version": sys.version,
        "python_chess_version": getattr(chess, "__version__", "unknown"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "openings": [
            {**opening, "fen": apply_opening(opening["moves"]).fen()}
            for opening in openings
        ],
    }
    with open(paths["metadata"], "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    for row in summaries:
        print(
            f"{row['base_time_per_move']:.3f}s: "
            f"completed={row['completed_games']}/{row['scheduled_games']}, "
            f"W={row['v2_wins']}, D={row['draws']}, L={row['v2_losses']}, "
            f"score rate={row['completed_score_rate']:.3f}, "
            f"truncated={row['truncated_games']}, "
            f"incomplete nodes V2/Fixed="
            f"{row['v2_incomplete_node_fraction']:.3f}/"
            f"{row['fixed_incomplete_node_fraction']:.3f}"
        )

    print(f"\nWall time={time.perf_counter() - experiment_start:.2f}s")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return games, summaries, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-budgets",
        type=float,
        nargs="+",
        default=list(FORMAL_BASE_BUDGETS),
        help="Nominal seconds per move (formal frozen default: 0.10 0.20)",
    )
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--opening-limit", type=int)
    parser.add_argument(
        "--output-dir",
        default="results/formal_search_aware_v2",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run two 100 ms validation games with a 40-ply cap",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pilot:
        run_search_aware_experiment(
            base_budgets=[0.10],
            max_plies=40,
            max_depth=min(args.max_depth, 6),
            opening_limit=1,
            output_dir=args.output_dir,
            experiment_name="search_aware_v2_pilot",
        )
    else:
        run_search_aware_experiment(
            base_budgets=args.base_budgets,
            max_plies=args.max_plies,
            max_depth=args.max_depth,
            opening_limit=args.opening_limit,
            output_dir=args.output_dir,
        )
