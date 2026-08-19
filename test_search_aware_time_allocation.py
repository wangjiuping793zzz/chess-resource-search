import time
import unittest

import chess

from search_aware_time_limited import (
    allocate_search_aware_targets,
    classify_search_stability,
    estimate_next_iteration_time,
    find_best_move_time_managed,
)


def iteration(
    depth,
    move,
    score,
    gap,
    seconds=0.01,
    nodes=100,
    root_move_count=20,
):
    return {
        "depth": depth,
        "best_move_uci": move,
        "score": score,
        "iteration_time": seconds,
        "iteration_nodes": nodes,
        "root_move_count": root_move_count,
        "observed_root_score_gap": gap,
    }


class SearchAwareTimeAllocationTests(unittest.TestCase):
    def test_time_targets_use_frozen_multipliers(self):
        targets = allocate_search_aware_targets(0.10, 8.0, 80)
        self.assertAlmostEqual(targets["stable"], 0.08)
        self.assertAlmostEqual(targets["uncertain"], 0.10)
        self.assertAlmostEqual(targets["unstable"], 0.20)

    def test_targets_protect_future_minimum_reserve(self):
        targets = allocate_search_aware_targets(0.10, 0.88, 10)
        # Nine future moves reserve 9 * 0.08 = 0.72 seconds, leaving 0.16 now.
        self.assertAlmostEqual(targets["stable"], 0.08)
        self.assertAlmostEqual(targets["uncertain"], 0.10)
        self.assertAlmostEqual(targets["unstable"], 0.16)

    def test_all_three_instability_signals_are_unstable(self):
        records = [
            iteration(1, "g1f3", 20, 80),
            iteration(2, "b1c3", 120, 10),
        ]
        result = classify_search_stability(records)
        self.assertEqual(result["stability_class"], "unstable")
        self.assertTrue(result["best_move_changed"])
        self.assertEqual(result["instability_signal_count"], 3)

    def test_same_move_small_change_wide_gap_is_stable(self):
        records = [
            iteration(1, "g1f3", 20, 80),
            iteration(2, "g1f3", 35, 90),
        ]
        result = classify_search_stability(records)
        self.assertEqual(result["stability_class"], "stable")
        self.assertFalse(result["best_move_changed"])
        self.assertEqual(result["absolute_score_change"], 15)
        self.assertEqual(result["instability_signal_count"], 0)

    def test_two_instability_signals_are_uncertain(self):
        records = [
            iteration(1, "g1f3", 20, 80),
            iteration(2, "b1c3", 120, 90),
        ]
        result = classify_search_stability(records)
        self.assertEqual(result["stability_class"], "uncertain")
        self.assertEqual(result["instability_signal_count"], 2)

    def test_first_depth_remains_uncertain_without_cross_depth_history(self):
        result = classify_search_stability(
            [iteration(1, "g1f3", 20, 10)]
        )
        self.assertEqual(result["stability_class"], "uncertain")
        self.assertEqual(result["instability_signal_count"], 0)

    def test_forced_move_is_stable(self):
        result = classify_search_stability(
            [iteration(1, "e1f1", 0, None, root_move_count=1)]
        )
        self.assertEqual(result["stability_class"], "stable")
        self.assertEqual(result["reason"], "forced_move")

    def test_next_iteration_prediction_combines_time_and_node_growth(self):
        records = [
            iteration(1, "g1f3", 20, 80, seconds=0.01, nodes=20),
            iteration(2, "g1f3", 25, 90, seconds=0.03, nodes=100),
        ]
        prediction = estimate_next_iteration_time(records)
        self.assertIsNotNone(prediction)
        # Geometric mean of time growth 3 and node growth 5, with 1.10 safety.
        expected_growth = (3.0 * 5.0) ** 0.5
        self.assertAlmostEqual(
            prediction["predicted_seconds"],
            0.03 * expected_growth * 1.10,
        )
        self.assertAlmostEqual(prediction["clamped_growth"], expected_growth)

    def test_fixed_search_returns_legal_move_and_restores_board(self):
        board = chess.Board()
        fen_before = board.fen()
        start = time.perf_counter()
        move, _score, stats = find_best_move_time_managed(
            board,
            strategy="fixed",
            base_time_per_move=0.05,
            remaining_clock=4.0,
            remaining_move_capacity=80,
            max_depth=6,
            decision_start=start,
        )
        self.assertIn(move, board.legal_moves)
        self.assertEqual(board.fen(), fen_before)
        self.assertGreaterEqual(stats["completed_depth"], 1)
        self.assertGreater(stats["nodes"], 0)

    def test_search_aware_search_returns_diagnostics(self):
        board = chess.Board()
        fen_before = board.fen()
        start = time.perf_counter()
        move, _score, stats = find_best_move_time_managed(
            board,
            strategy="search_aware",
            base_time_per_move=0.10,
            remaining_clock=8.0,
            remaining_move_capacity=80,
            max_depth=6,
            decision_start=start,
        )
        self.assertIn(move, board.legal_moves)
        self.assertEqual(board.fen(), fen_before)
        self.assertGreaterEqual(stats["completed_depth"], 1)
        self.assertIn(stats["stability_class"], {"stable", "uncertain", "unstable"})
        self.assertIn(
            stats["stop_reason"],
            {
                "deadline_timeout",
                "max_depth_completed",
                "predicted_next_iteration_exceeds_target",
                "selected_time_target_reached",
            },
        )
        self.assertEqual(
            stats["incomplete_iteration_nodes"],
            stats["nodes"] - sum(stats["iteration_nodes"]),
        )

    def test_exhausted_clock_uses_legal_depth_zero_fallback(self):
        board = chess.Board()
        move, score, stats = find_best_move_time_managed(
            board,
            strategy="search_aware",
            base_time_per_move=0.10,
            remaining_clock=0.0,
            remaining_move_capacity=1,
        )
        self.assertIn(move, board.legal_moves)
        self.assertIsNone(score)
        self.assertEqual(stats["completed_depth"], 0)
        self.assertTrue(stats["timed_out"])


if __name__ == "__main__":
    unittest.main()
