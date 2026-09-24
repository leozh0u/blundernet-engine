"""Search behaviour that has to hold whatever the network weights are.

Priors come from an untrained net here, so these tests only assert what the
tree itself guarantees: legality, visit accounting, and that a forced mate
beats whatever the policy happened to like.
"""

import chess
import pytest
import torch

from blundernet import mcts
from blundernet.model import BlunderNet

# Back-rank mate in one: Ra8 is checkmate, everything else is not.
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BlunderNet().eval()


def test_visits_only_legal_moves(model):
    board = chess.Board()

    visits = mcts.search(board, model, simulations=32)

    assert set(visits) == set(board.legal_moves)


def test_visit_counts_add_up(model):
    board = chess.Board()
    simulations = 48

    visits = mcts.search(board, model, simulations=simulations)

    assert sum(visits.values()) == simulations


def test_search_leaves_the_board_untouched(model):
    board = chess.Board(MATE_IN_ONE)
    before = board.fen()

    mcts.search(board, model, simulations=16)

    assert board.fen() == before


def test_finds_mate_in_one(model):
    """The mating move returns a terminal loss for the opponent every visit, so
    it should out-visit anything the untrained policy prefers."""
    board = chess.Board(MATE_IN_ONE)

    move = mcts.best_move(board, model, simulations=160)

    assert board.is_checkmate() is False
    board.push(move)
    assert board.is_checkmate()


def test_best_move_is_legal_from_a_middlegame(model):
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5")

    assert mcts.best_move(board, model, simulations=32) in board.legal_moves


def test_temperature_sampling_returns_a_legal_move(model):
    board = chess.Board()

    move = mcts.best_move(board, model, simulations=32, temperature=1.0)

    assert move in board.legal_moves


def test_terminal_value_reads_from_the_side_to_move():
    checkmated = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 b - - 1 1")
    checkmated.push_san("Kh8")
    checkmated.push_san("Ra8")

    assert checkmated.is_checkmate()
    assert mcts._terminal_value(checkmated) == -1.0
    assert mcts._terminal_value(chess.Board()) is None


def test_stalemate_is_a_draw():
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

    assert stalemate.is_stalemate()
    assert mcts._terminal_value(stalemate) == 0.0


@pytest.fixture(params=["python", "cpp"])
def search_backend(request):
    from blundernet import mcts_cpp

    if request.param == "cpp":
        if not mcts_cpp.AVAILABLE:
            pytest.skip("C++ extension not built")
        return mcts_cpp
    return mcts


@pytest.mark.parametrize("fen", [
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # checkmate
    "7k/8/6K1/8/8/8/8/8 w - - 0 1",  # insufficient material
])
def test_finished_game_has_no_search_moves(search_backend, model, fen):
    board = chess.Board(fen)
    assert search_backend.search(board, model, simulations=4) == {}
    with pytest.raises(ValueError, match="game is over"):
        search_backend.best_move(board, model, simulations=4)


def test_automatic_repetition_ends_search(search_backend, model):
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 4:
        board.push_uci(move)
    assert board.is_fivefold_repetition()
    assert search_backend.search(board, model, simulations=4) == {}


def test_positive_search_budget_required(search_backend, model):
    with pytest.raises(ValueError, match="simulations"):
        search_backend.search(chess.Board(), model, simulations=0)


@pytest.mark.parametrize("simulations,temperature", [(1, 1.0), (32, 0.001)])
def test_sampling_extreme_budgets_and_temperatures(search_backend, model, simulations, temperature):
    board = chess.Board()
    assert search_backend.best_move(
        board, model, simulations=simulations, temperature=temperature
    ) in board.legal_moves


def test_cpp_rejects_empty_batches(model):
    from blundernet import mcts_cpp

    if not mcts_cpp.AVAILABLE:
        pytest.skip("C++ extension not built")
    with pytest.raises(ValueError, match="batch_size"):
        mcts_cpp.search(chess.Board(), model, batch_size=0)


def test_search_preserves_history_in_simulations(search_backend, model, monkeypatch):
    board = chess.Board()
    board.push_uci("g1f3")
    original_copy = chess.Board.copy
    copied_stacks = []

    def record_copy(self, *, stack=True):
        copied = original_copy(self, stack=stack)
        copied_stacks.append(list(copied.move_stack))
        return copied

    monkeypatch.setattr(chess.Board, "copy", record_copy)
    search_backend.search(board, model, simulations=4)
    assert copied_stacks
    assert all(stack == board.move_stack for stack in copied_stacks)
