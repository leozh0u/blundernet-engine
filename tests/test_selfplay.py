import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from blundernet.encode import PLANES, POLICY_SIZE

spec = importlib.util.spec_from_file_location(
    "selfplay", Path(__file__).resolve().parents[1] / "scripts" / "selfplay.py"
)
selfplay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selfplay)


def test_selfplay_preserves_probability_when_promotions_share_an_index():
    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(POLICY_SIZE))

        def forward(self, x):
            return self.logits.expand(len(x), -1), torch.zeros(len(x))

    model = Uniform()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    samples = [(
        np.zeros((PLANES, 8, 8), dtype=np.float32),
        np.array([123, 123], dtype=np.int64),
        np.array([0.25, 0.75], dtype=np.float32),
        0.0,
    )]
    loss = selfplay.train_on_selfplay(model, opt, {"steps": 0, "samples_seen": 0}, samples)
    assert loss == pytest.approx(np.log(POLICY_SIZE))
