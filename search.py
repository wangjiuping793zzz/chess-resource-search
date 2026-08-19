import chess

from evaluation import evaluate_board, get_terminal_score


def _validate_depth(depth: int) -> None:
    """Require root searches to use at least one ply."""
    if depth < 1:
        raise ValueError("Search depth must be at least 1.")


def _recursive_position_score(
    board: chess.Board,
    depth: int,
):
    """
    Return a score when a recursive node should stop, otherwise return None.

    Search-tree nodes use claim_draw=False. This still recognises automatic
    terminal outcomes, including checkmate, stalemate, insufficient material,
    the seventy-five-move rule and fivefold repetition, but avoids repeatedly
    checking claimable threefold and fifty-move draws at every node.

    Claimable draws are handled at the root and by the match runner.
    """

    terminal_score = get_terminal_score(
        board,
        claim_draw=False,
    )

    if terminal_score is not None:
        return terminal_score

    if depth <= 0:
        return evaluate_board(
            board,
            claim_draw=False,
        )

    return None


def order_moves(board: chess.Board, moves):
    """
    Order legal moves to improve Alpha-Beta pruning efficiency.

    Priority:
    1. Captures
    2. Promotions
    3. Checks
    4. Other moves
    """

    def move_score(move: chess.Move) -> int:
        score = 0

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            attacking_piece = board.piece_at(move.from_square)

            if captured_piece is not None and attacking_piece is not None:
                score += (
                    10 * captured_piece.piece_type
                    - attacking_piece.piece_type
                )
            else:
                # Covers special captures such as en passant.
                score += 10

        if move.promotion is not None:
            score += 100

        board.push(move)
        try:
            if board.is_check():
                score += 50
        finally:
            board.pop()

        return score

    return sorted(
        list(moves),
        key=move_score,
        reverse=True,
    )


def alpha_beta_ordered(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    stats: dict,
) -> int:
    """Alpha-Beta pruning with move ordering."""

    stats["nodes"] += 1

    stopping_score = _recursive_position_score(
        board,
        depth,
    )
    if stopping_score is not None:
        return stopping_score

    ordered_moves = order_moves(
        board,
        board.legal_moves,
    )

    if maximizing_player:
        best_score = -float("inf")

        for move in ordered_moves:
            board.push(move)
            try:
                score = alpha_beta_ordered(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    stats,
                )
            finally:
                board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, best_score)

            if beta <= alpha:
                stats["cutoffs"] += 1
                break

        return best_score

    best_score = float("inf")

    for move in ordered_moves:
        board.push(move)
        try:
            score = alpha_beta_ordered(
                board,
                depth - 1,
                alpha,
                beta,
                True,
                stats,
            )
        finally:
            board.pop()

        best_score = min(best_score, score)
        beta = min(beta, best_score)

        if beta <= alpha:
            stats["cutoffs"] += 1
            break

    return best_score


def find_best_move_alpha_beta_ordered(
    board: chess.Board,
    depth: int,
):
    """Find the best move using Alpha-Beta pruning with move ordering."""

    _validate_depth(depth)

    stats = {
        "nodes": 0,
        "cutoffs": 0,
    }

    root_terminal_score = get_terminal_score(
        board,
        claim_draw=True,
    )
    if root_terminal_score is not None:
        return None, root_terminal_score, stats

    best_move = None
    alpha = -float("inf")
    beta = float("inf")
    ordered_moves = order_moves(
        board,
        board.legal_moves,
    )

    if board.turn == chess.WHITE:
        best_score = -float("inf")

        for move in ordered_moves:
            board.push(move)
            try:
                score = alpha_beta_ordered(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    stats,
                )
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

    else:
        best_score = float("inf")

        for move in ordered_moves:
            board.push(move)
            try:
                score = alpha_beta_ordered(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    stats,
                )
            finally:
                board.pop()

            if score < best_score:
                best_score = score
                best_move = move

            beta = min(beta, best_score)

    return best_move, best_score, stats


def minimax(
    board: chess.Board,
    depth: int,
    maximizing_player: bool,
    stats: dict,
) -> int:
    """Basic Minimax algorithm without pruning."""

    stats["nodes"] += 1

    stopping_score = _recursive_position_score(
        board,
        depth,
    )
    if stopping_score is not None:
        return stopping_score

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        best_score = -float("inf")

        for move in legal_moves:
            board.push(move)
            try:
                score = minimax(
                    board,
                    depth - 1,
                    False,
                    stats,
                )
            finally:
                board.pop()

            best_score = max(best_score, score)

        return best_score

    best_score = float("inf")

    for move in legal_moves:
        board.push(move)
        try:
            score = minimax(
                board,
                depth - 1,
                True,
                stats,
            )
        finally:
            board.pop()

        best_score = min(best_score, score)

    return best_score


def find_best_move_minimax(
    board: chess.Board,
    depth: int,
):
    """Find the best move for the current player using basic Minimax."""

    _validate_depth(depth)

    stats = {
        "nodes": 0,
    }

    root_terminal_score = get_terminal_score(
        board,
        claim_draw=True,
    )
    if root_terminal_score is not None:
        return None, root_terminal_score, stats

    best_move = None
    legal_moves = list(board.legal_moves)

    if board.turn == chess.WHITE:
        best_score = -float("inf")

        for move in legal_moves:
            board.push(move)
            try:
                score = minimax(
                    board,
                    depth - 1,
                    False,
                    stats,
                )
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

    else:
        best_score = float("inf")

        for move in legal_moves:
            board.push(move)
            try:
                score = minimax(
                    board,
                    depth - 1,
                    True,
                    stats,
                )
            finally:
                board.pop()

            if score < best_score:
                best_score = score
                best_move = move

    return best_move, best_score, stats


def alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    stats: dict,
) -> int:
    """Minimax search with Alpha-Beta pruning."""

    stats["nodes"] += 1

    stopping_score = _recursive_position_score(
        board,
        depth,
    )
    if stopping_score is not None:
        return stopping_score

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        best_score = -float("inf")

        for move in legal_moves:
            board.push(move)
            try:
                score = alpha_beta(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    stats,
                )
            finally:
                board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, best_score)

            if beta <= alpha:
                stats["cutoffs"] += 1
                break

        return best_score

    best_score = float("inf")

    for move in legal_moves:
        board.push(move)
        try:
            score = alpha_beta(
                board,
                depth - 1,
                alpha,
                beta,
                True,
                stats,
            )
        finally:
            board.pop()

        best_score = min(best_score, score)
        beta = min(beta, best_score)

        if beta <= alpha:
            stats["cutoffs"] += 1
            break

    return best_score


def find_best_move_alpha_beta(
    board: chess.Board,
    depth: int,
):
    """Find the best move using Minimax with Alpha-Beta pruning."""

    _validate_depth(depth)

    stats = {
        "nodes": 0,
        "cutoffs": 0,
    }

    root_terminal_score = get_terminal_score(
        board,
        claim_draw=True,
    )
    if root_terminal_score is not None:
        return None, root_terminal_score, stats

    best_move = None
    alpha = -float("inf")
    beta = float("inf")
    legal_moves = list(board.legal_moves)

    if board.turn == chess.WHITE:
        best_score = -float("inf")

        for move in legal_moves:
            board.push(move)
            try:
                score = alpha_beta(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    stats,
                )
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

    else:
        best_score = float("inf")

        for move in legal_moves:
            board.push(move)
            try:
                score = alpha_beta(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    stats,
                )
            finally:
                board.pop()

            if score < best_score:
                best_score = score
                best_move = move

            beta = min(beta, best_score)

    return best_move, best_score, stats
