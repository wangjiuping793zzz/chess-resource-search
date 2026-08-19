import chess
import time

from search import (
    find_best_move_minimax,
    find_best_move_alpha_beta,
    find_best_move_alpha_beta_ordered
)

from metrics import calculate_effective_branching_factor


def run_search_test(board: chess.Board, depth: int):
    print("Current board:")
    print(board)

    print(f"\nSearch depth: {depth}")

    # Minimax
    start = time.time()
    minimax_move, minimax_score, minimax_stats = find_best_move_minimax(board, depth)
    minimax_time = time.time() - start

    # Alpha-Beta
    start = time.time()
    ab_move, ab_score, ab_stats = find_best_move_alpha_beta(board, depth)
    ab_time = time.time() - start

    # Alpha-Beta with Move Ordering
    start = time.time()
    ordered_move, ordered_score, ordered_stats = find_best_move_alpha_beta_ordered(board, depth)
    ordered_time = time.time() - start

    print("\n=== Minimax ===")
    print(f"Best move: {minimax_move}")
    print(f"Best score: {minimax_score}")
    print(f"Nodes searched: {minimax_stats['nodes']}")
    print(f"Effective branching factor: {calculate_effective_branching_factor(minimax_stats['nodes'], depth):.2f}")
    print(f"Search time: {minimax_time:.4f} seconds")

    print("\n=== Alpha-Beta ===")
    print(f"Best move: {ab_move}")
    print(f"Best score: {ab_score}")
    print(f"Nodes searched: {ab_stats['nodes']}")
    print(f"Cutoffs: {ab_stats['cutoffs']}")
    print(f"Effective branching factor: {calculate_effective_branching_factor(ab_stats['nodes'], depth):.2f}")
    print(f"Search time: {ab_time:.4f} seconds")

    print("\n=== Alpha-Beta + Move Ordering ===")
    print(f"Best move: {ordered_move}")
    print(f"Best score: {ordered_score}")
    print(f"Nodes searched: {ordered_stats['nodes']}")
    print(f"Cutoffs: {ordered_stats['cutoffs']}")
    print(f"Effective branching factor: {calculate_effective_branching_factor(ordered_stats['nodes'], depth):.2f}")
    print(f"Search time: {ordered_time:.4f} seconds")


def main():
    board = chess.Board()
    depth = 3
    run_search_test(board, depth)


if __name__ == "__main__":
    main()