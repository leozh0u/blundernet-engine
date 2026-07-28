"""Shape and range checks on the network, plus the size budget it was designed to."""

import chess
import numpy as np
import torch

from blundernet.encode import PLANES, POLICY_SIZE, encode_board
from blundernet.evaluate import move_accuracy
from blundernet.model import BlunderNet


def test_forward_shapes():
    model = BlunderNet().eval()
    x = torch.zeros(4, PLANES, 8, 8)

    with torch.no_grad():
        policy, value = model(x)

    assert policy.shape == (4, POLICY_SIZE)
    assert value.shape == (4,)


def test_value_head_is_bounded():
    model = BlunderNet().eval()
    x = torch.randn(8, PLANES, 8, 8)

    with torch.no_grad():
        _, value = model(x)

    assert torch.all(value >= -1) and torch.all(value <= 1)


def test_accepts_a_real_position():
    model = BlunderNet().eval()
    x = torch.from_numpy(encode_board(chess.Board())).unsqueeze(0)

    with torch.no_grad():
        policy, value = model(x)

    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


def test_stays_within_the_size_budget():
    """Small on purpose: every scheduled run trains on free CI hardware."""
    model = BlunderNet()
    params = sum(p.numel() for p in model.parameters())
    trunk = sum(p.numel() for p in model.blocks.parameters())

    assert 2_000_000 < params < 3_000_000
    # Most of the weight is the policy head's 512 -> 4096 projection, not the
    # residual trunk. Worth knowing before trying to shrink the model.
    assert trunk < params / 4


def test_move_accuracy_counts_a_known_hit():
    """A model that always ranks index 5 first should score 1.0 on a target of 5."""

    class Fixed(torch.nn.Module):
        def eval(self):
            return self

        def forward(self, x):
            logits = torch.zeros(len(x), POLICY_SIZE)
            logits[:, 5] = 10.0
            logits[:, 6] = 5.0
            return logits, torch.zeros(len(x))

    X = np.zeros((3, PLANES, 8, 8), dtype=np.float32)

    hit = move_accuracy(Fixed(), X, np.array([5, 5, 5]))
    near = move_accuracy(Fixed(), X, np.array([6, 6, 6]))

    assert hit == {"top1": 1.0, "top3": 1.0, "eval_positions": 3}
    assert near["top1"] == 0.0
    assert near["top3"] == 1.0
