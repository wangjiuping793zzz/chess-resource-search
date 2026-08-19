import unittest

import chess

from evaluation import MATE_SCORE, evaluate_board
from search import (
    find_best_move_alpha_beta,
    find_best_move_alpha_beta_ordered,
    find_best_move_minimax,
    order_moves,
)


SEARCH_FUNCTIONS = {
    "minimax": find_best_move_minimax,
    "alpha_beta": find_best_move_alpha_beta,
    "ordered_alpha_beta": find_best_move_alpha_beta_ordered,
}


def board_from_san_moves(moves):
    """Create a board by applying a sequence of SAN moves."""
    board = chess.Board()

    for move in moves:
        board.push_san(move)

    return board


class EvaluationValidationTests(unittest.TestCase):
    """Validate score direction and terminal-position handling."""

    def test_starting_position_is_balanced(self):
        board = chess.Board()
        self.assertEqual(evaluate_board(board), 0)

    def test_white_material_advantage_is_positive(self):
        board = chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 0 1")
        self.assertGreater(evaluate_board(board), 0)

    def test_black_material_advantage_is_negative(self):
        board = chess.Board("kq6/8/8/8/8/8/8/7K b - - 0 1")
        self.assertLess(evaluate_board(board), 0)

    def test_white_checkmate_win_is_positive_mate_score(self):
        # Black to move is checkmated.
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_checkmate())
        self.assertEqual(evaluate_board(board), MATE_SCORE)

    def test_black_checkmate_win_is_negative_mate_score(self):
        # White to move is checkmated.
        board = chess.Board("8/8/8/8/8/6k1/6q1/7K w - - 0 1")
        self.assertTrue(board.is_checkmate())
        self.assertEqual(evaluate_board(board), -MATE_SCORE)

    def test_stalemate_is_zero(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_stalemate())
        self.assertEqual(evaluate_board(board), 0)

    def test_insufficient_material_is_zero(self):
        board = chess.Board("8/8/8/8/8/8/6k1/K7 w - - 0 1")
        self.assertTrue(board.is_insufficient_material())
        self.assertEqual(evaluate_board(board), 0)

    def test_claimable_threefold_repetition_is_zero(self):
        board = chess.Board()

        repetition_sequence = [
            "Nf3", "Nf6", "Ng1", "Ng8",
            "Nf3", "Nf6", "Ng1", "Ng8",
        ]

        for move in repetition_sequence:
            board.push_san(move)

        self.assertTrue(board.can_claim_threefold_repetition())
        self.assertEqual(evaluate_board(board), 0)

    def test_claimable_fifty_move_draw_is_zero(self):
        board = chess.Board("8/8/8/8/8/8/R6k/K7 w - - 100 51")
        self.assertTrue(board.can_claim_fifty_moves())
        self.assertEqual(evaluate_board(board), 0)


class SearchValidationTests(unittest.TestCase):
    """Validate equivalence, legality, node counting and board restoration."""

    def setUp(self):
        self.positions = {
            "starting_position": chess.Board(),
            "black_to_move_opening": board_from_san_moves(["e4"]),
            "early_opening": board_from_san_moves(
                ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
            ),
            "simple_endgame": chess.Board(
                "8/8/8/4k3/8/4K3/8/4R3 w - - 0 1"
            ),
        }

    def run_all_searches(self, board, depth):
        """Run every search from the same board and verify restoration."""
        results = {}
        original_fen = board.fen()
        original_stack = list(board.move_stack)

        for name, search_function in SEARCH_FUNCTIONS.items():
            move, score, stats = search_function(board, depth)

            self.assertEqual(
                board.fen(),
                original_fen,
                msg=f"{name} changed the board FEN at depth {depth}.",
            )
            self.assertEqual(
                list(board.move_stack),
                original_stack,
                msg=f"{name} changed the board move stack at depth {depth}.",
            )
            self.assertIn(
                move,
                board.legal_moves,
                msg=f"{name} returned an illegal move at depth {depth}.",
            )
            self.assertGreater(
                stats["nodes"],
                0,
                msg=f"{name} reported no searched nodes at depth {depth}.",
            )

            results[name] = {
                "move": move,
                "score": score,
                "stats": stats,
            }

        return results

    def test_algorithms_return_equal_scores_at_depths_one_and_two(self):
        for position_name, board in self.positions.items():
            for depth in (1, 2):
                with self.subTest(position=position_name, depth=depth):
                    results = self.run_all_searches(board, depth)

                    minimax_score = results["minimax"]["score"]
                    alpha_beta_score = results["alpha_beta"]["score"]
                    ordered_score = results["ordered_alpha_beta"]["score"]

                    self.assertEqual(minimax_score, alpha_beta_score)
                    self.assertEqual(minimax_score, ordered_score)

    def test_depth_one_nodes_equal_root_legal_move_count(self):
        for position_name, board in self.positions.items():
            expected_nodes = board.legal_moves.count()

            for algorithm_name, search_function in SEARCH_FUNCTIONS.items():
                with self.subTest(
                    position=position_name,
                    algorithm=algorithm_name,
                ):
                    _, _, stats = search_function(board, 1)
                    self.assertEqual(stats["nodes"], expected_nodes)

    def test_alpha_beta_never_searches_more_nodes_than_minimax(self):
        for position_name, board in self.positions.items():
            for depth in (1, 2):
                with self.subTest(position=position_name, depth=depth):
                    _, _, minimax_stats = find_best_move_minimax(board, depth)
                    _, _, alpha_beta_stats = find_best_move_alpha_beta(
                        board,
                        depth,
                    )
                    _, _, ordered_stats = (
                        find_best_move_alpha_beta_ordered(board, depth)
                    )

                    self.assertLessEqual(
                        alpha_beta_stats["nodes"],
                        minimax_stats["nodes"],
                    )
                    self.assertLessEqual(
                        ordered_stats["nodes"],
                        minimax_stats["nodes"],
                    )

    def test_order_moves_does_not_change_board(self):
        board = board_from_san_moves(
            ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
        )
        original_fen = board.fen()
        original_stack = list(board.move_stack)
        legal_moves = list(board.legal_moves)

        ordered_moves = order_moves(board, legal_moves)

        self.assertEqual(board.fen(), original_fen)
        self.assertEqual(list(board.move_stack), original_stack)
        self.assertEqual(set(ordered_moves), set(legal_moves))

    def test_terminal_root_returns_no_move_and_exact_score(self):
        positions = [
            (
                chess.Board(
                    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
                ),
                MATE_SCORE,
            ),
            (
                chess.Board(
                    "8/8/8/8/8/6k1/6q1/7K w - - 0 1"
                ),
                -MATE_SCORE,
            ),
            (
                chess.Board(
                    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
                ),
                0,
            ),
        ]

        for board, expected_score in positions:
            for algorithm_name, search_function in SEARCH_FUNCTIONS.items():
                with self.subTest(
                    fen=board.fen(),
                    algorithm=algorithm_name,
                ):
                    move, score, stats = search_function(board, 2)
                    self.assertIsNone(move)
                    self.assertEqual(score, expected_score)
                    self.assertEqual(stats["nodes"], 0)

    def test_invalid_root_depth_raises_value_error(self):
        board = chess.Board()

        for algorithm_name, search_function in SEARCH_FUNCTIONS.items():
            with self.subTest(algorithm=algorithm_name):
                with self.assertRaises(ValueError):
                    search_function(board, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
