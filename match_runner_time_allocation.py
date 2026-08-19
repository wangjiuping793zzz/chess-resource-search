"""Formal fixed-versus-dynamic time-allocation chess experiment.

Default formal design:
    3 nominal per-move budgets x 12 openings x 2 colours = 72 games.

Both players receive the same initial total clock in every paired game.  The
fixed policy requests the nominal amount each move; the dynamic policy requests
0.6x, 1.0x, or 1.8x according to pre-search position volatility.  All engine
decisions use the same evaluator and time-limited ordered Alpha-Beta search.
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
from time_limited_search import (
    FUTURE_RESERVE_FACTOR,
    QUIET_THRESHOLD,
    TIME_MULTIPLIERS,
    VOLATILE_THRESHOLD,
    VOLATILITY_WEIGHTS,
    allocate_decision_time,
    find_best_move_iterative,
    measure_position_volatility,
)


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


def score_for_dynamic(result: str, dynamic_colour: str) -> Optional[float]:
    if result == "*":
        return None
    if result == "1/2-1/2":
        return 0.5
    if dynamic_colour == "white":
        return 1.0 if result == "1-0" else 0.0
    return 1.0 if result == "0-1" else 0.0


def _new_side_totals() -> Dict[str, object]:
    return {
        "moves": 0,
        "nodes": 0,
        "cutoffs": 0,
        "allocated_time": 0.0,
        "decision_time": 0.0,
        "search_time": 0.0,
        "analysis_time": 0.0,
        "completed_depth_sum": 0,
        "maximum_completed_depth": 0,
        "depth_zero_fallbacks": 0,
        "deadline_overruns": 0,
        "quiet_positions": 0,
        "normal_positions": 0,
        "volatile_positions": 0,
    }


def play_game(
    opening: Dict[str, object],
    base_time_per_move: float,
    dynamic_colour: str,
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
        dynamic_colour: "dynamic",
        "black" if dynamic_colour == "white" else "white": "fixed",
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
            1, player_move_capacity - int(side_totals["moves"])
        )
        decision_start = time.perf_counter()

        if strategy == "dynamic":
            volatility_start = time.perf_counter()
            volatility = measure_position_volatility(board)
            analysis_time = time.perf_counter() - volatility_start
            allocation = allocate_decision_time(
                strategy,
                base_time_per_move,
                clock_before,
                remaining_move_capacity,
                volatility,
            )
            search_start = time.perf_counter()
            deadline = decision_start + allocation
            move, score, stats = find_best_move_iterative(
                board, deadline, max_depth=max_depth
            )
            search_end = time.perf_counter()
            search_time = search_end - search_start
            decision_time = search_end - decision_start
            logging_volatility_time = 0.0
        else:
            allocation = allocate_decision_time(
                strategy,
                base_time_per_move,
                clock_before,
                remaining_move_capacity,
            )
            search_start = time.perf_counter()
            deadline = decision_start + allocation
            move, score, stats = find_best_move_iterative(
                board, deadline, max_depth=max_depth
            )
            search_end = time.perf_counter()
            search_time = search_end - search_start
            decision_time = search_end - decision_start
            analysis_time = 0.0

            # Volatility is still logged for analysis, but the fixed policy does
            # not use it and the diagnostic work is not charged to its clock.
            logging_start = time.perf_counter()
            volatility = measure_position_volatility(board)
            logging_volatility_time = time.perf_counter() - logging_start

        if move is None:
            raise RuntimeError(f"No move in non-terminal game {game_id}")
        if move not in board.legal_moves:
            raise RuntimeError(
                f"Illegal move {move} in game {game_id}; FEN={board.fen()}"
            )

        san = board.san(move)
        clock_after = max(0.0, clock_before - decision_time)
        clocks[side] = clock_after
        overrun = max(0.0, decision_time - allocation)
        completed_depth = int(stats["completed_depth"])
        volatility_class = str(volatility["volatility_class"])

        side_totals["moves"] = int(side_totals["moves"]) + 1
        side_totals["nodes"] = int(side_totals["nodes"]) + int(stats["nodes"])
        side_totals["cutoffs"] = int(side_totals["cutoffs"]) + int(
            stats["cutoffs"]
        )
        side_totals["allocated_time"] = float(
            side_totals["allocated_time"]
        ) + allocation
        side_totals["decision_time"] = float(side_totals["decision_time"]) + (
            decision_time
        )
        side_totals["search_time"] = float(side_totals["search_time"]) + (
            search_time
        )
        side_totals["analysis_time"] = float(side_totals["analysis_time"]) + (
            analysis_time
        )
        side_totals["completed_depth_sum"] = int(
            side_totals["completed_depth_sum"]
        ) + completed_depth
        side_totals["maximum_completed_depth"] = max(
            int(side_totals["maximum_completed_depth"]), completed_depth
        )
        side_totals["depth_zero_fallbacks"] = int(
            side_totals["depth_zero_fallbacks"]
        ) + int(completed_depth == 0)
        side_totals["deadline_overruns"] = int(
            side_totals["deadline_overruns"]
        ) + int(overrun > 0)
        class_key = f"{volatility_class}_positions"
        side_totals[class_key] = int(side_totals[class_key]) + 1

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
                "dynamic_colour": dynamic_colour,
                "fen_before_move": board.fen(),
                "move_uci": move.uci(),
                "move_san": san,
                "score_white_perspective": "" if score is None else score,
                "initial_clock": initial_clock,
                "clock_before": clock_before,
                "allocated_decision_time": allocation,
                "policy_analysis_time": analysis_time,
                "search_time": search_time,
                "decision_time_charged": decision_time,
                "deadline_overrun": overrun,
                "clock_after": clock_after,
                "completed_depth": completed_depth,
                "attempted_depth": stats["attempted_depth"],
                "timed_out": stats["timed_out"],
                "nodes": stats["nodes"],
                "cutoffs": stats["cutoffs"],
                "iteration_nodes": "|".join(
                    str(value) for value in stats["iteration_nodes"]
                ),
                "volatility_score": volatility["volatility_score"],
                "volatility_class": volatility_class,
                "time_multiplier": (
                    volatility["time_multiplier"]
                    if strategy == "dynamic"
                    else 1.0
                ),
                "in_check": volatility["in_check"],
                "legal_move_count": volatility["legal_move_count"],
                "capture_move_count": volatility["capture_move_count"],
                "checking_move_count": volatility["checking_move_count"],
                "absolute_evaluation_swing": volatility[
                    "absolute_evaluation_swing"
                ],
                "diagnostic_volatility_time_not_charged": (
                    logging_volatility_time
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

    dynamic_score = score_for_dynamic(result, dynamic_colour)
    fixed_colour = "black" if dynamic_colour == "white" else "white"
    dynamic_totals = totals[dynamic_colour]
    fixed_totals = totals[fixed_colour]

    def role_fields(role: str, data: Dict[str, object], colour: str):
        moves = int(data["moves"])
        return {
            f"{role}_colour": colour,
            f"{role}_move_count": moves,
            f"{role}_total_nodes": data["nodes"],
            f"{role}_average_nodes_per_move": safe_average(
                float(data["nodes"]), moves
            ),
            f"{role}_total_cutoffs": data["cutoffs"],
            f"{role}_total_allocated_time": data["allocated_time"],
            f"{role}_total_decision_time": data["decision_time"],
            f"{role}_average_decision_time_per_move": safe_average(
                float(data["decision_time"]), moves
            ),
            f"{role}_average_completed_depth": safe_average(
                float(data["completed_depth_sum"]), moves
            ),
            f"{role}_maximum_completed_depth": data["maximum_completed_depth"],
            f"{role}_depth_zero_fallbacks": data["depth_zero_fallbacks"],
            f"{role}_deadline_overruns": data["deadline_overruns"],
            f"{role}_quiet_positions": data["quiet_positions"],
            f"{role}_normal_positions": data["normal_positions"],
            f"{role}_volatile_positions": data["volatile_positions"],
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
        "dynamic_score": "" if dynamic_score is None else dynamic_score,
        "engine_plies_played": engine_plies,
        "total_plies_from_start": len(opening["moves"]) + engine_plies,
        "engine_moves_uci": " ".join(engine_uci),
        "engine_moves_san": " ".join(engine_san),
        "full_game_san": " ".join(list(opening["moves"]) + engine_san),
        "final_fen": board.fen(),
        "final_evaluation_white_perspective": final_evaluation,
        "game_wall_time": game_wall_time,
        **role_fields("dynamic", dynamic_totals, dynamic_colour),
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
                "dynamic_wins": 0,
                "draws": 0,
                "dynamic_losses": 0,
                "completed_score": 0.0,
                "dynamic_white_completed_games": 0,
                "dynamic_white_score": 0.0,
                "dynamic_black_completed_games": 0,
                "dynamic_black_score": 0.0,
                "dynamic_moves": 0,
                "fixed_moves": 0,
                "dynamic_nodes": 0,
                "fixed_nodes": 0,
                "dynamic_decision_time": 0.0,
                "fixed_decision_time": 0.0,
                "dynamic_completed_depth_weighted_sum": 0.0,
                "fixed_completed_depth_weighted_sum": 0.0,
                "dynamic_clock_used": 0.0,
                "fixed_clock_used": 0.0,
                "dynamic_depth_zero_fallbacks": 0,
                "fixed_depth_zero_fallbacks": 0,
                "dynamic_deadline_overruns": 0,
                "fixed_deadline_overruns": 0,
            },
        )

        data["scheduled_games"] += 1
        for role in ("dynamic", "fixed"):
            moves = int(row[f"{role}_move_count"])
            data[f"{role}_moves"] += moves
            data[f"{role}_nodes"] += int(row[f"{role}_total_nodes"])
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
            data[f"{role}_deadline_overruns"] += int(
                row[f"{role}_deadline_overruns"]
            )

        if not row["completed_game"]:
            data["truncated_games"] += 1
            continue

        data["completed_games"] += 1
        score = float(row["dynamic_score"])
        data["completed_score"] += score
        if score == 1.0:
            data["dynamic_wins"] += 1
        elif score == 0.5:
            data["draws"] += 1
        else:
            data["dynamic_losses"] += 1

        colour = row["dynamic_colour"]
        data[f"dynamic_{colour}_completed_games"] += 1
        data[f"dynamic_{colour}_score"] += score

    summaries = []
    for budget in sorted(grouped):
        data = grouped[budget]
        summaries.append(
            {
                **data,
                "completed_score_rate": safe_average(
                    float(data["completed_score"]), int(data["completed_games"])
                ),
                "score_rate_if_truncations_count_as_half": safe_average(
                    float(data["completed_score"])
                    + 0.5 * int(data["truncated_games"]),
                    int(data["scheduled_games"]),
                ),
                "dynamic_white_score_rate": safe_average(
                    float(data["dynamic_white_score"]),
                    int(data["dynamic_white_completed_games"]),
                ),
                "dynamic_black_score_rate": safe_average(
                    float(data["dynamic_black_score"]),
                    int(data["dynamic_black_completed_games"]),
                ),
                "dynamic_average_nodes_per_move": safe_average(
                    float(data["dynamic_nodes"]), int(data["dynamic_moves"])
                ),
                "fixed_average_nodes_per_move": safe_average(
                    float(data["fixed_nodes"]), int(data["fixed_moves"])
                ),
                "dynamic_average_decision_time_per_move": safe_average(
                    float(data["dynamic_decision_time"]),
                    int(data["dynamic_moves"]),
                ),
                "fixed_average_decision_time_per_move": safe_average(
                    float(data["fixed_decision_time"]), int(data["fixed_moves"])
                ),
                "dynamic_average_completed_depth": safe_average(
                    float(data["dynamic_completed_depth_weighted_sum"]),
                    int(data["dynamic_moves"]),
                ),
                "fixed_average_completed_depth": safe_average(
                    float(data["fixed_completed_depth_weighted_sum"]),
                    int(data["fixed_moves"]),
                ),
            }
        )
    return summaries


def run_time_allocation_experiment(
    base_budgets: Optional[List[float]] = None,
    max_plies: int = 160,
    max_depth: int = 8,
    opening_limit: Optional[int] = None,
    output_dir: str = "results",
    experiment_name: str = "formal_time_allocation_matches",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, str]]:
    if base_budgets is None:
        base_budgets = [0.05, 0.10, 0.20]
    if any(value <= 0 for value in base_budgets):
        raise ValueError("All base budgets must be positive")

    openings = OPENING_LINES[:opening_limit] if opening_limit else OPENING_LINES
    validate_opening_lines(openings)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"time_allocation_{timestamp}"
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
            for dynamic_colour in ("white", "black"):
                game_number += 1
                budget_ms = int(round(budget * 1000))
                game_id = (
                    f"t{budget_ms}ms_{opening['name']}_dynamic_{dynamic_colour}"
                )
                print(f"\n[{game_number}/{total_games}] {game_id}")
                game, game_moves = play_game(
                    opening=opening,
                    base_time_per_move=budget,
                    dynamic_colour=dynamic_colour,
                    experiment_timestamp=timestamp,
                    game_id=game_id,
                    max_plies=max_plies,
                    max_depth=max_depth,
                )
                games.append(game)
                moves.extend(game_moves)
                save_csv(paths["game_checkpoint"], games)
                save_csv(paths["move_checkpoint"], moves)

                score = game["dynamic_score"] if game["dynamic_score"] != "" else "excluded"
                print(
                    f"result={game['result']}; termination={game['termination_reason']}; "
                    f"dynamic score={score}; plies={game['engine_plies_played']}; "
                    f"wall={game['game_wall_time']:.2f}s"
                )

    summaries = summarise(games)
    save_csv(paths["game_results"], games)
    save_csv(paths["move_results"], moves)
    save_csv(paths["summary"], summaries)

    metadata = {
        "experiment_timestamp": timestamp,
        "experiment_name": experiment_name,
        "research_comparison": "fixed versus volatility-aware dynamic allocation",
        "base_time_per_move_seconds": base_budgets,
        "opening_count": len(openings),
        "games_per_budget": len(openings) * 2,
        "scheduled_games": total_games,
        "max_engine_plies": max_plies,
        "max_iterative_depth": max_depth,
        "player_move_capacity": math.ceil(max_plies / 2),
        "initial_clock_formula": "base_time_per_move * ceil(max_engine_plies / 2)",
        "fairness_control": (
            "Both players receive the same initial total clock; the same "
            "evaluator, ordered Alpha-Beta implementation, maximum depth, "
            "opening, and colour-pairing rules are used."
        ),
        "fixed_policy": "min(base_time_per_move, remaining_clock)",
        "dynamic_multipliers": TIME_MULTIPLIERS,
        "future_reserve_factor": FUTURE_RESERVE_FACTOR,
        "dynamic_policy_overhead": (
            "Volatility analysis is included in the dynamic player's decision "
            "time and deducted from its clock."
        ),
        "volatility_formula": {
            "weights": VOLATILITY_WEIGHTS,
            "capture_normalisation": "min(capture_moves / 3, 1)",
            "checking_move_normalisation": "min(checking_moves / 2, 1)",
            "evaluation_swing_normalisation": "min(abs(eval_now - eval_before_last_move) / 200, 1)",
            "quiet_if_score_below": QUIET_THRESHOLD,
            "volatile_if_score_at_least": VOLATILE_THRESHOLD,
            "otherwise": "normal",
        },
        "search_method": (
            "Iterative deepening with ordered Alpha-Beta; an interrupted "
            "iteration is discarded and the last completed iteration is used."
        ),
        "truncation_policy": (
            'Games reaching max_engine_plies use result="*" and are excluded '
            "from the primary completed-game score rate."
        ),
        "primary_strength_metric": "Dynamic-policy score rate over completed games.",
        "primary_resource_metric": "Actual decision time deducted from equal clocks.",
        "secondary_metrics": [
            "nodes",
            "completed iterative-deepening depth",
            "deadline overrun",
            "volatility class distribution",
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
            f"W={row['dynamic_wins']}, D={row['draws']}, "
            f"L={row['dynamic_losses']}, "
            f"score rate={row['completed_score_rate']:.3f}, "
            f"truncated={row['truncated_games']}"
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
        default=[0.05, 0.10, 0.20],
        help="Nominal seconds per move (formal default: 0.05 0.10 0.20)",
    )
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--opening-limit", type=int)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 2 short validation games instead of the formal design",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pilot:
        run_time_allocation_experiment(
            base_budgets=[0.02],
            max_plies=20,
            max_depth=min(args.max_depth, 6),
            opening_limit=1,
            output_dir=args.output_dir,
            experiment_name="time_allocation_pilot",
        )
    else:
        run_time_allocation_experiment(
            base_budgets=args.base_budgets,
            max_plies=args.max_plies,
            max_depth=args.max_depth,
            opening_limit=args.opening_limit,
            output_dir=args.output_dir,
        )
