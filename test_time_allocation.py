import time
import unittest

import chess

from time_limited_search import (
    allocate_decision_time,
    find_best_move_iterative,
    measure_position_volatility,
)


class TimeAllocationTests(unittest.TestCase):
    def test_starting_position_is_quiet(self):
        features = measure_position_volatility(chess.Board())
        self.assertEqual(features["volatility_class"], "quiet")
        self.assertEqual(features["time_multiplier"], 0.60)

    def test_check_position_is_volatile(self):
        board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
        features = measure_position_volatility(board)
        self.assertTrue(features["in_check"])
        self.assertEqual(features["volatility_class"], "volatile")
        self.assertEqual(features["time_multiplier"], 1.80)

    def test_fixed_and_dynamic_allocations_respect_clock(self):
        fixed = allocate_decision_time("fixed", 0.10, 2.0, 20)
        quiet = allocate_decision_time(
            "dynamic",
            0.10,
            2.0,
            20,
            {"time_multiplier": 0.60},
        )
        volatile = allocate_decision_time(
            "dynamic",
            0.10,
            2.0,
            20,
            {"time_multiplier": 1.80},
        )
        self.assertAlmostEqual(fixed, 0.10)
        self.assertAlmostEqual(quiet, 0.06)
        self.assertAlmostEqual(volatile, 0.18)
        self.assertLessEqual(volatile, 2.0)

    def test_dynamic_policy_preserves_quiet_budget_for_future_moves(self):
        base_time = 0.05
        remaining_clock = 1.0
        allocations = []

        # A worst-case run of 20 consecutive volatile positions must not spend
        # the future quiet-policy reserve early.
        for remaining_moves in range(20, 0, -1):
            allocation = allocate_decision_time(
                "dynamic",
                base_time,
                remaining_clock,
                remaining_moves,
                {"time_multiplier": 1.80},
            )
            allocations.append(allocation)
            remaining_clock -= allocation

        for allocation in allocations[:-1]:
            self.assertGreaterEqual(allocation + 1e-12, base_time * 0.60)
        self.assertAlmostEqual(sum(allocations), 1.0)
        self.assertAlmostEqual(remaining_clock, 0.0)

    def test_iterative_search_returns_legal_move_and_restores_board(self):
        board = chess.Board()
        fen_before = board.fen()
        move, _score, stats = find_best_move_iterative(
            board, time.perf_counter() + 0.05, max_depth=6
        )
        self.assertIn(move, board.legal_moves)
        self.assertEqual(board.fen(), fen_before)
        self.assertGreaterEqual(stats["completed_depth"], 1)
        self.assertGreater(stats["nodes"], 0)

    def test_expired_deadline_uses_legal_depth_zero_fallback(self):
        board = chess.Board()
        fen_before = board.fen()
        move, score, stats = find_best_move_iterative(
            board, time.perf_counter() - 1.0, max_depth=6
        )
        self.assertIn(move, board.legal_moves)
        self.assertIsNone(score)
        self.assertEqual(stats["completed_depth"], 0)
        self.assertTrue(stats["timed_out"])
        self.assertEqual(board.fen(), fen_before)


if __name__ == "__main__":
    unittest.main()
