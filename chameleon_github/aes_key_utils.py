"""AES 单字节攻击的候选评分与评估；不依赖 CNN。"""
import numpy as np
import torch
from aes_constants import AES_SBOX, HW

def candidate_scores(log_probabilities, plaintexts, encoding):
    """输入不包含真实密钥：枚举 256 个假设并逐次累积对数似然。"""
    candidates = np.arange(256, dtype=np.uint8)
    intermediate = AES_SBOX[plaintexts[:, None] ^ candidates[None, :]]
    if encoding == "hw":
        intermediate = HW[intermediate]
        prior = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1]) / 256
        log_probabilities = log_probabilities - np.log(prior)
    return log_probabilities[np.arange(len(plaintexts))[:, None], intermediate].cumsum(
        axis=0
    )


def evaluate_ranks(logits, plaintexts, true_keys, encoding, seed=123, repeats=1):
    """每个密钥独立攻击；真实密钥仅在评分完成后用于检查排名。"""
    logp = torch.log_softmax(torch.as_tensor(logits), dim=1).numpy()
    rng = np.random.default_rng(seed)
    n = min(np.count_nonzero(true_keys == key) for key in np.unique(true_keys))
    curves, rows = [], []
    for key in np.unique(true_keys):
        ids = np.flatnonzero(true_keys == key)
        for repetition in range(repeats):
            selected = rng.permutation(ids)[:n]
            scores = candidate_scores(logp[selected], plaintexts[selected], encoding)
            # 并列时保守处理，避免相同分数被 argsort 偶然排成第 1。
            correct = scores[:, int(key)]
            ranks = np.sum(scores >= correct[:, None], axis=1)
            curves.append(ranks)
            rows.append(
                dict(
                    true_key=int(key),
                    repetition=repetition,
                    best_guess=int(scores[-1].argmax()),
                    final_rank=int(ranks[-1]),
                )
            )
    ranks = np.stack(curves)
    return dict(
        attack_encryptions=n,
        independent_keys=len(np.unique(true_keys)),
        repeats=repeats,
        mean_rank=ranks.mean(0).tolist(),
        success_rate=(ranks == 1).mean(0).tolist(),
        per_key=rows,
    )


