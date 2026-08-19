import csv
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime

import chess

from metrics import (
    calculate_approximate_effective_branching_factor,
    calculate_node_reduction,
)
from search import (
    find_best_move_alpha_beta,
    find_best_move_alpha_beta_ordered,
    find_best_move_minimax,
)


ALGORITHM_RUNNERS = {
    "minimax": find_best_move_minimax,
    "alphabeta": find_best_move_alpha_beta,
    "ordered": find_best_move_alpha_beta_ordered,
}


def board_from_san_moves(moves):
    """Create a board by applying a sequence of SAN moves."""

    board = chess.Board()

    for move in moves:
        board.push_san(move)

    return board


def get_test_positions():
    """Return the representative chess positions used in the experiment."""

    early_opening_moves = [
        "e4", "e5",
        "Nf3", "Nc6",
        "Bb5", "a6",
    ]

    middlegame_moves = [
        "e4", "e5",
        "Nf3", "Nc6",
        "Bb5", "a6",
        "Ba4", "Nf6",
        "O-O", "Be7",
        "Re1", "b5",
        "Bb3", "d6",
        "c3", "O-O",
        "h3",
    ]

    return [
        {
            "name": "starting_position",
            "description": "Standard chess starting position",
            "source_type": "standard_start",
            "setup_moves_san": [],
            "board": chess.Board(),
        },
        {
            "name": "early_opening",
            "description": "Early Ruy Lopez opening position",
            "source_type": "san_sequence",
            "setup_moves_san": early_opening_moves,
            "board": board_from_san_moves(early_opening_moves),
        },
        {
            "name": "middlegame",
            "description": "Ruy Lopez middlegame-like position",
            "source_type": "san_sequence",
            "setup_moves_san": middlegame_moves,
            "board": board_from_san_moves(middlegame_moves),
        },
        {
            "name": "simple_endgame",
            "description": "Simple rook-and-kings endgame position",
            "source_type": "fen",
            "setup_moves_san": [],
            "board": chess.Board(
                "8/8/8/4k3/8/4K3/8/4R3 w - - 0 1"
            ),
        },
    ]


def run_search_once(board, depth, search_function):
    """
    Run one timed search and verify that the search restores the board.

    The returned node count follows the project convention: recursive
    successor positions are counted, while the root position is excluded.
    """

    original_fen = board.fen()
    original_stack = list(board.move_stack)

    start_time = time.perf_counter()
    move, score, stats = search_function(board, depth)
    elapsed = time.perf_counter() - start_time

    if board.fen() != original_fen:
        raise RuntimeError(
            "Search changed the board FEN. "
            f"Before: {original_fen}; after: {board.fen()}"
        )

    if list(board.move_stack) != original_stack:
        raise RuntimeError("Search changed the board move stack.")

    if move is None:
        raise RuntimeError(
            "Search returned no move for a non-terminal experiment position. "
            f"FEN: {original_fen}"
        )

    if move not in board.legal_moves:
        raise RuntimeError(
            f"Search returned illegal move {move}. FEN: {original_fen}"
        )

    return {
        "best_move": str(move),
        "score": score,
        "nodes": int(stats.get("nodes", 0)),
        "cutoffs": int(stats.get("cutoffs", 0)),
        "time": elapsed,
    }


def summarise_repeated_results(algorithm_name, repeated_results):
    """Validate deterministic outputs and summarise repeated runtimes."""

    if not repeated_results:
        raise ValueError(f"No results recorded for {algorithm_name}.")

    move_values = {row["best_move"] for row in repeated_results}
    score_values = {row["score"] for row in repeated_results}
    node_values = {row["nodes"] for row in repeated_results}
    cutoff_values = {row["cutoffs"] for row in repeated_results}

    if len(move_values) != 1:
        raise RuntimeError(
            f"{algorithm_name} returned inconsistent best moves "
            f"across repeated runs: {sorted(move_values)}"
        )

    if len(score_values) != 1:
        raise RuntimeError(
            f"{algorithm_name} returned inconsistent scores "
            f"across repeated runs: {sorted(score_values)}"
        )

    if len(node_values) != 1:
        raise RuntimeError(
            f"{algorithm_name} returned inconsistent node counts "
            f"across repeated runs: {sorted(node_values)}"
        )

    if len(cutoff_values) != 1:
        raise RuntimeError(
            f"{algorithm_name} returned inconsistent cutoff counts "
            f"across repeated runs: {sorted(cutoff_values)}"
        )

    times = [row["time"] for row in repeated_results]

    return {
        "best_move": repeated_results[0]["best_move"],
        "score": repeated_results[0]["score"],
        "nodes": repeated_results[0]["nodes"],
        "cutoffs": repeated_results[0]["cutoffs"],
        "time_median": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "time_runs": times,
    }


def run_repeated_searches(board, depth, runtime_repeats):
    """
    Run all three algorithms repeatedly.

    The execution order is rotated between repetitions to reduce systematic
    bias caused by always timing one algorithm first.
    """

    if runtime_repeats < 1:
        raise ValueError("runtime_repeats must be at least 1.")

    algorithm_names = list(ALGORITHM_RUNNERS)
    repeated = {name: [] for name in algorithm_names}

    for repeat_index in range(runtime_repeats):
        rotation = repeat_index % len(algorithm_names)
        run_order = (
            algorithm_names[rotation:]
            + algorithm_names[:rotation]
        )

        for algorithm_name in run_order:
            gc.collect()

            board_copy = board.copy(stack=True)
            search_function = ALGORITHM_RUNNERS[algorithm_name]

            result = run_search_once(
                board_copy,
                depth,
                search_function,
            )
            repeated[algorithm_name].append(result)

    summaries = {
        name: summarise_repeated_results(name, rows)
        for name, rows in repeated.items()
    }

    scores = {
        name: summary["score"]
        for name, summary in summaries.items()
    }

    if len(set(scores.values())) != 1:
        raise RuntimeError(
            "Search algorithms returned different depth-limited scores. "
            f"Depth={depth}, FEN={board.fen()}, scores={scores}"
        )

    return summaries


def format_time_runs(times):
    """Store all repeated runtime measurements in one CSV field."""

    return "|".join(f"{value:.9f}" for value in times)


def build_result_row(
    position,
    depth,
    runtime_repeats,
    summaries,
    experiment_timestamp,
):
    """Build one CSV row for a position-depth combination."""

    board = position["board"]

    minimax = summaries["minimax"]
    alphabeta = summaries["alphabeta"]
    ordered = summaries["ordered"]

    minimax_aebf = calculate_approximate_effective_branching_factor(
        minimax["nodes"],
        depth,
    )
    alphabeta_aebf = calculate_approximate_effective_branching_factor(
        alphabeta["nodes"],
        depth,
    )
    ordered_aebf = calculate_approximate_effective_branching_factor(
        ordered["nodes"],
        depth,
    )

    return {
        "experiment_timestamp": experiment_timestamp,
        "position_name": position["name"],
        "position_description": position["description"],
        "position_source_type": position["source_type"],
        "setup_moves_san": " ".join(position["setup_moves_san"]),
        "position_fen": board.fen(),
        "side_to_move": (
            "white" if board.turn == chess.WHITE else "black"
        ),
        "legal_moves_at_root": board.legal_moves.count(),
        "depth": depth,
        "runtime_repeats": runtime_repeats,
        "score_equivalence_validated": True,

        "minimax_best_move": minimax["best_move"],
        "minimax_score": minimax["score"],
        "minimax_nodes": minimax["nodes"],
        "minimax_time_median": minimax["time_median"],
        "minimax_time_min": minimax["time_min"],
        "minimax_time_max": minimax["time_max"],
        "minimax_time_runs": format_time_runs(
            minimax["time_runs"]
        ),
        "minimax_approximate_effective_branching_factor": minimax_aebf,

        "alphabeta_best_move": alphabeta["best_move"],
        "alphabeta_score": alphabeta["score"],
        "alphabeta_nodes": alphabeta["nodes"],
        "alphabeta_cutoffs": alphabeta["cutoffs"],
        "alphabeta_time_median": alphabeta["time_median"],
        "alphabeta_time_min": alphabeta["time_min"],
        "alphabeta_time_max": alphabeta["time_max"],
        "alphabeta_time_runs": format_time_runs(
            alphabeta["time_runs"]
        ),
        "alphabeta_approximate_effective_branching_factor": (
            alphabeta_aebf
        ),

        "ordered_best_move": ordered["best_move"],
        "ordered_score": ordered["score"],
        "ordered_nodes": ordered["nodes"],
        "ordered_cutoffs": ordered["cutoffs"],
        "ordered_time_median": ordered["time_median"],
        "ordered_time_min": ordered["time_min"],
        "ordered_time_max": ordered["time_max"],
        "ordered_time_runs": format_time_runs(
            ordered["time_runs"]
        ),
        "ordered_approximate_effective_branching_factor": ordered_aebf,

        "alphabeta_node_reduction_percent": calculate_node_reduction(
            minimax["nodes"],
            alphabeta["nodes"],
        ),
        "ordered_node_reduction_percent": calculate_node_reduction(
            minimax["nodes"],
            ordered["nodes"],
        ),
        "ordering_extra_node_reduction_percent": calculate_node_reduction(
            alphabeta["nodes"],
            ordered["nodes"],
        ),
    }


def save_results_to_csv(results, timestamp):
    """Save fixed-depth results to a timestamped CSV file."""

    os.makedirs("results", exist_ok=True)

    file_path = (
        f"results/fixed_depth_final_results_{timestamp}.csv"
    )

    fieldnames = list(results[0].keys())

    with open(
        file_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    return file_path


def save_experiment_metadata(
    positions,
    max_depth,
    runtime_repeats,
    timestamp,
):
    """Save software environment and experiment settings as JSON."""

    os.makedirs("results", exist_ok=True)

    metadata = {
        "experiment_timestamp": timestamp,
        "experiment_name": "fixed_depth_search_complexity",
        "max_depth": max_depth,
        "runtime_repeats": runtime_repeats,
        "runtime_summary": "median",
        "timer": "time.perf_counter",
        "node_counting_convention": (
            "Recursive successor positions are counted; "
            "the root position is excluded."
        ),
        "approximate_effective_branching_factor": (
            "nodes ** (1 / depth)"
        ),
        "algorithm_run_order": (
            "Rotated across repetitions to reduce systematic timing bias."
        ),
        "python_version": sys.version,
        "python_chess_version": getattr(
            chess,
            "__version__",
            "unknown",
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "positions": [
            {
                "name": position["name"],
                "description": position["description"],
                "source_type": position["source_type"],
                "setup_moves_san": position["setup_moves_san"],
                "fen": position["board"].fen(),
            }
            for position in positions
        ],
    }

    file_path = (
        f"results/fixed_depth_metadata_{timestamp}.json"
    )

    with open(file_path, mode="w", encoding="utf-8") as json_file:
        json.dump(
            metadata,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    return file_path


def run_fixed_depth_experiments(
    max_depth: int = 4,
    runtime_repeats: int = 3,
):
    """
    Run the formal fixed-depth search-complexity experiment.

    For a quick local check before the formal run, use runtime_repeats=1.
    The formal dataset should use runtime_repeats=3 or more.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")

    if runtime_repeats < 1:
        raise ValueError("runtime_repeats must be at least 1.")

    positions = get_test_positions()
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for position in positions:
        board = position["board"]

        if board.outcome(claim_draw=True) is not None:
            raise ValueError(
                "Experiment position is already terminal: "
                f"{position['name']}"
            )

        print("\n====================================")
        print(f"Position: {position['name']}")
        print(f"Description: {position['description']}")
        print(f"FEN: {board.fen()}")
        print(f"Side to move: {'White' if board.turn else 'Black'}")
        print(f"Legal moves: {board.legal_moves.count()}")
        print("====================================")

        for depth in range(1, max_depth + 1):
            print(
                f"\nRunning {position['name']} at depth {depth} "
                f"with {runtime_repeats} timing repetition(s)..."
            )

            summaries = run_repeated_searches(
                board,
                depth,
                runtime_repeats,
            )

            result_row = build_result_row(
                position,
                depth,
                runtime_repeats,
                summaries,
                timestamp,
            )
            results.append(result_row)

            print(
                f"  Minimax nodes: {result_row['minimax_nodes']}"
            )
            print(
                f"  Alpha-Beta nodes: "
                f"{result_row['alphabeta_nodes']}"
            )
            print(
                f"  Ordered Alpha-Beta nodes: "
                f"{result_row['ordered_nodes']}"
            )
            print(
                f"  Alpha-Beta reduction: "
                f"{result_row['alphabeta_node_reduction_percent']:.2f}%"
            )
            print(
                f"  Ordering extra reduction: "
                f"{result_row['ordering_extra_node_reduction_percent']:.2f}%"
            )

    csv_path = save_results_to_csv(results, timestamp)
    metadata_path = save_experiment_metadata(
        positions,
        max_depth,
        runtime_repeats,
        timestamp,
    )

    print(f"\nResults saved to: {csv_path}")
    print(f"Metadata saved to: {metadata_path}")

    return results, csv_path, metadata_path


def run_multiple_position_experiments(
    max_depth: int = 4,
    runtime_repeats: int = 3,
):
    """
    Backward-compatible wrapper for the earlier experiment function name.
    """

    return run_fixed_depth_experiments(
        max_depth=max_depth,
        runtime_repeats=runtime_repeats,
    )


if __name__ == "__main__":
    run_fixed_depth_experiments(
        max_depth=4,
        runtime_repeats=3,
    )
