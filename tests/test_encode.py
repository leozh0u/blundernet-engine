"""The encoding is the contract between chess and the network. If it drifts,
every metric in this repo silently measures something else."""

import chess
import numpy as np

from blundernet.encode import (
    PLANES,
    POLICY_SIZE,
    encode_board,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


def test_shape_and_binary():
    x = encode_board(chess.Board())

    assert x.shape == (PLANES, 8, 8)
    assert x.dtype == np.float32
    assert set(np.unique(x)) <= {0.0, 1.0}


def test_starting_position_piece_counts():
    x = encode_board(chess.Board())

    # Planes 0-5 are white pawn..king, 6-11 the same for black.
    assert x[0].sum() == 8  # white pawns
    assert x[5].sum() == 1  # white king
    assert x[6].sum() == 8  # black pawns
    assert x[11].sum() == 1  # black king
    assert x[:12].sum() == 32


def test_pieces_land_on_their_own_squares():
    board = chess.Board()
    x = encode_board(board)

    for square, piece in board.piece_map().items():
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        assert x[plane, square // 8, square % 8] == 1.0


def test_side_to_move_plane():
    board = chess.Board()
    assert encode_board(board)[12].all()

    board.push_san("e4")
    assert not encode_board(board)[12].any()


def test_castling_planes_clear_when_rights_are_lost():
    board = chess.Board()
    assert encode_board(board)[13:17].sum() == 4 * 64

    # Moving the white king gives up both white castling rights.
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Ke2")
    x = encode_board(board)

    assert not x[13].any()
    assert not x[14].any()
    assert x[15].all()
    assert x[16].all()


def test_en_passant_plane_marks_the_file():
    board = chess.Board()
    board.push_san("e4")
    x = encode_board(board)

    # e-file is index 4, and the plane marks a column, not a single square.
    assert x[17][:, 4].all()
    assert x[17].sum() == 8


def test_no_en_passant_leaves_the_plane_empty():
    assert not encode_board(chess.Board())[17].any()


def test_move_index_round_trips():
    board = chess.Board()

    for move in board.legal_moves:
        assert index_to_move(move_to_index(move), board) == move


def test_move_index_stays_in_the_policy():
    board = chess.Board()

    assert all(0 <= move_to_index(m) < POLICY_SIZE for m in board.legal_moves)


def test_promotion_is_restored_from_the_index():
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    push = chess.Move.from_uci("a7a8q")

    assert index_to_move(move_to_index(push), board) == push


def test_legal_mask_matches_python_chess():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    mask = legal_move_mask(board)

    assert mask.sum() == len(set(move_to_index(m) for m in board.legal_moves))
    assert all(mask[move_to_index(m)] for m in board.legal_moves)


def test_legal_mask_is_empty_at_checkmate():
    # Fool's mate.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")

    assert board.is_checkmate()
    assert not legal_move_mask(board).any()
