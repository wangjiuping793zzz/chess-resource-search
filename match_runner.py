import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import chess

from evaluation import evaluate_board
from search import (
    find_best_move_alpha_beta,
    find_best_move_alpha_beta_ordered,
)


# Every line has an even number of plies, so White is to move when engine
# play begins. Each line is played twice: challenger as White and as Black.
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


def apply_opening(moves):
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board


def validate_opening_lines(openings=None):
    if openings is None:
        openings = OPENING_LINES

    names = set()
    for opening in openings:
        name = opening["name"]
        moves = opening["moves"]

        if name in names:
            raise ValueError(f"Duplicate opening name: {name}")
        names.add(name)

        if len(moves) % 2 != 0:
            raise ValueError(f"{name} has an odd number of opening plies.")

        try:
            board = apply_opening(moves)
        except ValueError as error:
            raise ValueError(f"Invalid SAN sequence in {name}: {moves}") from error

        if board.turn != chess.WHITE:
            raise ValueError(f"{name} does not leave White to move.")

        if board.outcome(claim_draw=True) is not None:
            raise ValueError(f"{name} is already terminal.")

    return True


def choose_move(board, depth, algorithm):
    if algorithm == "alphabeta":
        return find_best_move_alpha_beta(board, depth)
    if algorithm == "ordered":
        return find_best_move_alpha_beta_ordered(board, depth)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def safe_average(total, count):
    return total / count if count else 0.0


def get_outcome(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None, None
    return outcome.result(), outcome.termination.name.lower()


def play_game(
    white_depth,
    black_depth,
    white_algorithm,
    black_algorithm,
    opening,
    experiment_timestamp,
    game_id,
    max_plies=160,
):
    board = apply_opening(opening["moves"])
    opening_fen = board.fen()

    totals = {
        "white_nodes": 0,
        "black_nodes": 0,
        "white_cutoffs": 0,
        "black_cutoffs": 0,
        "white_time": 0.0,
        "black_time": 0.0,
        "white_moves": 0,
        "black_moves": 0,
    }
    white_nodes_each = []
    black_nodes_each = []
    white_times_each = []
    black_times_each = []
    engine_uci = []
    engine_san = []
    engine_plies = 0

    wall_start = time.perf_counter()

    while board.outcome(claim_draw=True) is None and engine_plies < max_plies:
        side = "white" if board.turn == chess.WHITE else "black"
        depth = white_depth if side == "white" else black_depth
        algorithm = white_algorithm if side == "white" else black_algorithm

        start = time.perf_counter()
        move, score, stats = choose_move(board, depth, algorithm)
        elapsed = time.perf_counter() - start

        if move is None:
            raise RuntimeError(
                f"No move in non-terminal game {game_id}; FEN={board.fen()}"
            )
        if move not in board.legal_moves:
            raise RuntimeError(
                f"Illegal move {move} in game {game_id}; FEN={board.fen()}"
            )

        nodes = int(stats.get("nodes", 0))
        cutoffs = int(stats.get("cutoffs", 0))
        san = board.san(move)

        totals[f"{side}_nodes"] += nodes
        totals[f"{side}_cutoffs"] += cutoffs
        totals[f"{side}_time"] += elapsed
        totals[f"{side}_moves"] += 1

        if side == "white":
            white_nodes_each.append(nodes)
            white_times_each.append(elapsed)
        else:
            black_nodes_each.append(nodes)
            black_times_each.append(elapsed)

        engine_uci.append(move.uci())
        engine_san.append(san)
        board.push(move)
        engine_plies += 1

    wall_time = time.perf_counter() - wall_start
    result, termination = get_outcome(board)

    if result is None:
        completed = False
        result = "*"
        termination = "max_plies_truncation"
        final_eval = evaluate_board(board, claim_draw=False)
    else:
        completed = True
        outcome = board.outcome(claim_draw=True)
        final_eval = (
            0
            if outcome is not None and outcome.winner is None
            else evaluate_board(board, claim_draw=True)
        )

    return {
        "experiment_timestamp": experiment_timestamp,
        "game_id": game_id,
        "opening_name": opening["name"],
        "opening_family": opening["family"],
        "opening_moves_san": " ".join(opening["moves"]),
        "opening_fen": opening_fen,
        "opening_plies": len(opening["moves"]),
        "white_depth": white_depth,
        "black_depth": black_depth,
        "white_algorithm": white_algorithm,
        "black_algorithm": black_algorithm,
        "completed_game": completed,
        "result": result,
        "termination_reason": termination,
        "engine_plies_played": engine_plies,
        "total_plies_from_start": len(opening["moves"]) + engine_plies,
        "engine_moves_uci": " ".join(engine_uci),
        "engine_moves_san": " ".join(engine_san),
        "full_game_san": " ".join(opening["moves"] + engine_san),
        "final_fen": board.fen(),
        "final_evaluation_white_perspective": final_eval,
        "white_move_count": totals["white_moves"],
        "black_move_count": totals["black_moves"],
        "total_white_nodes": totals["white_nodes"],
        "total_black_nodes": totals["black_nodes"],
        "average_white_nodes_per_move": safe_average(
            totals["white_nodes"], totals["white_moves"]
        ),
        "average_black_nodes_per_move": safe_average(
            totals["black_nodes"], totals["black_moves"]
        ),
        "maximum_white_nodes_for_one_move": max(white_nodes_each, default=0),
        "maximum_black_nodes_for_one_move": max(black_nodes_each, default=0),
        "total_white_cutoffs": totals["white_cutoffs"],
        "total_black_cutoffs": totals["black_cutoffs"],
        "total_white_time": totals["white_time"],
        "total_black_time": totals["black_time"],
        "average_white_time_per_move": safe_average(
            totals["white_time"], totals["white_moves"]
        ),
        "average_black_time_per_move": safe_average(
            totals["black_time"], totals["black_moves"]
        ),
        "maximum_white_time_for_one_move": max(white_times_each, default=0.0),
        "maximum_black_time_for_one_move": max(black_times_each, default=0.0),
        "white_move_nodes": "|".join(map(str, white_nodes_each)),
        "black_move_nodes": "|".join(map(str, black_nodes_each)),
        "white_move_times": "|".join(f"{v:.9f}" for v in white_times_each),
        "black_move_times": "|".join(f"{v:.9f}" for v in black_times_each),
        "game_search_time": totals["white_time"] + totals["black_time"],
        "game_wall_time": wall_time,
    }


def challenger_score(result, colour):
    if result == "*":
        return None
    if result == "1/2-1/2":
        return 0.5
    if colour == "white":
        return 1.0 if result == "1-0" else 0.0
    if colour == "black":
        return 1.0 if result == "0-1" else 0.0
    raise ValueError("colour must be white or black")


def add_roles(game, challenger_depth, baseline_depth, colour):
    game["challenger_depth"] = challenger_depth
    game["baseline_depth"] = baseline_depth
    game["challenger_color"] = colour

    score = challenger_score(game["result"], colour)
    game["challenger_score"] = "" if score is None else score

    challenger = colour
    baseline = "black" if colour == "white" else "white"

    game["challenger_move_count"] = game[f"{challenger}_move_count"]
    game["baseline_move_count"] = game[f"{baseline}_move_count"]
    game["challenger_total_nodes"] = game[f"total_{challenger}_nodes"]
    game["baseline_total_nodes"] = game[f"total_{baseline}_nodes"]
    game["challenger_average_nodes_per_move"] = game[
        f"average_{challenger}_nodes_per_move"
    ]
    game["baseline_average_nodes_per_move"] = game[
        f"average_{baseline}_nodes_per_move"
    ]
    game["challenger_total_cutoffs"] = game[f"total_{challenger}_cutoffs"]
    game["baseline_total_cutoffs"] = game[f"total_{baseline}_cutoffs"]
    game["challenger_total_time"] = game[f"total_{challenger}_time"]
    game["baseline_total_time"] = game[f"total_{baseline}_time"]
    game["challenger_average_time_per_move"] = game[
        f"average_{challenger}_time_per_move"
    ]
    game["baseline_average_time_per_move"] = game[
        f"average_{baseline}_time_per_move"
    ]


def save_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarise(results):
    grouped = {}

    for row in results:
        depth = row["challenger_depth"]
        data = grouped.setdefault(
            depth,
            {
                "challenger_depth": depth,
                "baseline_depth": row["baseline_depth"],
                "scheduled_games": 0,
                "completed_games": 0,
                "truncated_games": 0,
                "challenger_wins": 0,
                "draws": 0,
                "challenger_losses": 0,
                "completed_score": 0.0,
                "challenger_total_nodes": 0,
                "baseline_total_nodes": 0,
                "challenger_total_moves": 0,
                "baseline_total_moves": 0,
                "challenger_total_search_time": 0.0,
                "baseline_total_search_time": 0.0,
                "total_game_wall_time": 0.0,
            },
        )

        data["scheduled_games"] += 1
        data["challenger_total_nodes"] += row["challenger_total_nodes"]
        data["baseline_total_nodes"] += row["baseline_total_nodes"]
        data["challenger_total_moves"] += row["challenger_move_count"]
        data["baseline_total_moves"] += row["baseline_move_count"]
        data["challenger_total_search_time"] += row["challenger_total_time"]
        data["baseline_total_search_time"] += row["baseline_total_time"]
        data["total_game_wall_time"] += row["game_wall_time"]

        if not row["completed_game"]:
            data["truncated_games"] += 1
            continue

        data["completed_games"] += 1
        score = float(row["challenger_score"])
        data["completed_score"] += score

        if score == 1.0:
            data["challenger_wins"] += 1
        elif score == 0.5:
            data["draws"] += 1
        else:
            data["challenger_losses"] += 1

    summaries = []
    for depth in sorted(grouped):
        data = grouped[depth]
        completed = data["completed_games"]
        scheduled = data["scheduled_games"]
        truncated = data["truncated_games"]

        summaries.append(
            {
                **data,
                "completed_score_rate": safe_average(
                    data["completed_score"], completed
                ),
                "score_rate_if_truncations_count_as_half": safe_average(
                    data["completed_score"] + 0.5 * truncated, scheduled
                ),
                "challenger_weighted_average_nodes_per_move": safe_average(
                    data["challenger_total_nodes"],
                    data["challenger_total_moves"],
                ),
                "baseline_weighted_average_nodes_per_move": safe_average(
                    data["baseline_total_nodes"], data["baseline_total_moves"]
                ),
                "challenger_weighted_average_time_per_move": safe_average(
                    data["challenger_total_search_time"],
                    data["challenger_total_moves"],
                ),
                "baseline_weighted_average_time_per_move": safe_average(
                    data["baseline_total_search_time"],
                    data["baseline_total_moves"],
                ),
            }
        )

    return summaries


def save_metadata(
    timestamp,
    openings,
    baseline_depth,
    challenger_depths,
    algorithm,
    max_plies,
    result_path,
    summary_path,
    checkpoint_path,
):
    metadata = {
        "experiment_timestamp": timestamp,
        "experiment_name": "formal_depth_strength_matches",
        "baseline_depth": baseline_depth,
        "challenger_depths": challenger_depths,
        "algorithm": algorithm,
        "opening_count": len(openings),
        "games_per_depth": len(openings) * 2,
        "scheduled_games": len(openings) * 2 * len(challenger_depths),
        "max_engine_plies": max_plies,
        "colour_control": (
            "Each opening is played twice: challenger as White and Black."
        ),
        "truncation_policy": (
            'Truncated games use result="*" and are excluded from the '
            "primary completed-game score rate."
        ),
        "primary_strength_metric": "Score rate over completed games.",
        "node_counting_convention": (
            "Recursive successor positions are counted; root excluded."
        ),
        "timer": "time.perf_counter",
        "result_csv": result_path,
        "summary_csv": summary_path,
        "checkpoint_csv": checkpoint_path,
        "python_version": sys.version,
        "python_chess_version": getattr(chess, "__version__", "unknown"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "openings": [
            {
                **opening,
                "fen": apply_opening(opening["moves"]).fen(),
            }
            for opening in openings
        ],
    }

    path = f"results/depth_strength_metadata_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return path


def run_depth_strength_experiment(
    baseline_depth=2,
    challenger_depths=None,
    algorithm="ordered",
    max_plies=160,
    opening_limit=None,
):
    if challenger_depths is None:
        challenger_depths = [2, 3, 4]

    openings = OPENING_LINES
    if opening_limit is not None:
        openings = openings[:opening_limit]

    validate_opening_lines(openings)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    checkpoint = f"results/depth_strength_checkpoint_{timestamp}.csv"
    final_csv = f"results/depth_strength_final_results_{timestamp}.csv"
    summary_csv = f"results/depth_strength_summary_{timestamp}.csv"

    total_games = len(openings) * 2 * len(challenger_depths)
    game_number = 0
    experiment_start = time.perf_counter()

    print(f"Validated {len(openings)} openings; scheduled games={total_games}.")

    for challenger_depth in challenger_depths:
        for opening in openings:
            for colour in ("white", "black"):
                game_number += 1
                game_id = (
                    f"d{challenger_depth}_{opening['name']}_challenger_{colour}"
                )
                print(f"\n[{game_number}/{total_games}] {game_id}")

                white_depth = (
                    challenger_depth if colour == "white" else baseline_depth
                )
                black_depth = (
                    baseline_depth if colour == "white" else challenger_depth
                )

                game = play_game(
                    white_depth,
                    black_depth,
                    algorithm,
                    algorithm,
                    opening,
                    timestamp,
                    game_id,
                    max_plies,
                )
                add_roles(game, challenger_depth, baseline_depth, colour)
                results.append(game)
                save_csv(checkpoint, results)

                score = (
                    game["challenger_score"]
                    if game["challenger_score"] != ""
                    else "excluded"
                )
                print(
                    f"result={game['result']}; "
                    f"termination={game['termination_reason']}; "
                    f"score={score}; "
                    f"engine plies={game['engine_plies_played']}; "
                    f"wall={game['game_wall_time']:.2f}s"
                )

    save_csv(final_csv, results)
    summaries = summarise(results)
    save_csv(summary_csv, summaries)

    metadata = save_metadata(
        timestamp,
        openings,
        baseline_depth,
        challenger_depths,
        algorithm,
        max_plies,
        final_csv,
        summary_csv,
        checkpoint,
    )

    print("\n=== Summary ===")
    for row in summaries:
        print(
            f"Depth {row['challenger_depth']} vs {row['baseline_depth']}: "
            f"completed={row['completed_games']}/{row['scheduled_games']}, "
            f"W={row['challenger_wins']}, D={row['draws']}, "
            f"L={row['challenger_losses']}, "
            f"score rate={row['completed_score_rate']:.3f}, "
            f"truncated={row['truncated_games']}"
        )

    print(f"\nWall time={time.perf_counter() - experiment_start:.2f}s")
    print(f"Final results: {final_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Metadata: {metadata}")
    print(f"Checkpoint: {checkpoint}")

    return results, summaries, final_csv, summary_csv, metadata, checkpoint


if __name__ == "__main__":
    run_depth_strength_experiment(
        baseline_depth=2,
        challenger_depths=[2, 3, 4],
        algorithm="ordered",
        max_plies=160,
    )
