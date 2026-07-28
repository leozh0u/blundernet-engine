"""Baseline opponents and the match bookkeeping every strength claim rests on."""

import random

import chess
import pytest

from blundernet.arena import elo_diff, match, material_greedy, play_game, random_mover


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(1234)


def test_random_mover_returns_a_legal_move():
    board = chess.Board()

    assert random_mover(board) in board.legal_moves


def test_material_greedy_takes_the_free_queen():
    # White rook on a1, black queen on a8, nothing in between.
    board = chess.Board("q5k1/8/8/8/8/8/8/R5K1 w - - 0 1")

    assert material_greedy(board) == chess.Move.from_uci("a1a8")


def test_material_greedy_prefers_mate_over_material():
    """Re8 is mate, Rxa2 wins a queen. A material-only search would grab the queen."""
    board = chess.Board("6k1/5ppp/8/8/8/8/q7/R3R1K1 w - - 0 1")

    move = material_greedy(board)
    board.push(move)

    assert board.is_checkmate()


def test_material_greedy_returns_a_legal_move_in_a_quiet_position():
    board = chess.Board()

    assert material_greedy(board) in board.legal_moves


def test_play_game_returns_a_result_string():
    result = play_game(random_mover, random_mover, max_plies=40)

    assert result in {"1-0", "0-1", "1/2-1/2"}


def test_match_accounting_adds_up():
    result = match(material_greedy, random_mover, games=4, max_plies=40)

    assert result["wins"] + result["draws"] + result["losses"] == result["games"] == 4
    assert result["score"] == pytest.approx(
        (result["wins"] + 0.5 * result["draws"]) / 4, abs=1e-3
    )


def test_elo_diff_is_zero_at_an_even_score():
    assert elo_diff(0.5) == 0.0


def test_elo_diff_signs_and_ordering():
    assert elo_diff(0.75) > 0
    assert elo_diff(0.25) < 0
    assert elo_diff(0.9) > elo_diff(0.6)


def test_elo_diff_clamps_a_perfect_score():
    """A clean sweep is a lower bound on strength, not infinite Elo."""
    perfect = elo_diff(1.0)

    assert perfect is not None
    assert perfect < 1300
