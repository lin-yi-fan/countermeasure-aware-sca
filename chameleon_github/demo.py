"""Synthetic smoke test; outputs are not experimental results."""
import numpy as np
import torch
from gate import Gate
from aes_constants import AES_SBOX
from aes_key_utils import candidate_scores, evaluate_ranks


def main():
    torch.manual_seed(7)
    model = Gate().eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 1000))
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()
    print("Untrained gate output shape:", tuple(logits.shape))

    # Toy leakage: deliberately favor the correct intermediate value.
    plaintexts = np.arange(32, dtype=np.uint8)
    keys = np.full(32, 42, dtype=np.uint8)
    target = AES_SBOX[plaintexts ^ keys]
    toy_logits = np.zeros((32, 256), dtype=np.float64)
    toy_logits[np.arange(32), target] = 5.0
    result = evaluate_ranks(toy_logits, plaintexts, keys, "identity", repeats=3)
    assert result["mean_rank"][-1] == 1.0
    assert result["success_rate"][-1] == 1.0

    # With no information all 256 candidates tie: conservative rank is 256.
    tied = evaluate_ranks(np.zeros_like(toy_logits), plaintexts, keys, "identity")
    assert tied["mean_rank"][-1] == 256.0
    assert tied["success_rate"][-1] == 0.0
    scores = candidate_scores(np.full((32, 256), -np.log(256)), plaintexts, "identity")
    assert scores.shape == (32, 256)
    print("Synthetic key-ranking and conservative-tie checks passed.")
    print("This demo does not reproduce the paper's reported metrics.")


if __name__ == "__main__":
    main()
