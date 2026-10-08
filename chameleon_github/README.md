# Countermeasure-Aware Expert Routing for AES Side-Channel Analysis

Code accompanying the paper **Countermeasure-Aware Expert Routing for AES Side-Channel Analysis** by Zhiwei He.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Chameleon data interface

Supply a local Chameleon HDF5 file and a trace ID present in that file:

```bash
python chameleon_data.py --h5 /path/to/chameleon_base.h5 --condition BASE --trace-id 0 --limit 100 --output features.npz
```

Replace the example path and trace ID with your own. Supported condition labels are `BASE`, `RD`, `DFS`, and `MRP`. Expected HDF5 paths:

```text
metadata/ciphers/ciphers_<id>/plaintexts    compound field p
metadata/ciphers/ciphers_<id>/key           compound field k
metadata/pinpoints/pinpoints_<id>           fields start, end
data/traces/trace_<id>
```

The loader selects eligible annotated AES operations, applies a third-order 125 kHz high-pass filter with 2,000-sample guards, and averages 100-sample windows over the first 100,000 samples. Output `x` has shape `[N, 1000]`. The default sampling rate is 125 MS/s; use `--sampling-rate` if your acquisition differs. `--byte-index` defaults to zero. `--limit` is a maximum; fewer eligible operations may be returned.

The NPZ also contains `condition`, `plaintexts`, `true_keys`, `event`, and `trace_id`. Preserve source-file identity when merging several files, since trace IDs can repeat between files. Select training/validation/test groups before fitting normalization. Keys are evaluation metadata and must not be supplied to the gate.

## Gate interface

```python
import numpy as np
import torch
from gate import Gate

z = np.load("features.npz")
# Arrays fitted on your training split and saved alongside trained weights:
mean = np.load("train_mean.npy")
std = np.load("train_std.npy")
x = ((z["x"] - mean) / np.maximum(std, 1e-6)).astype("float32")
model = Gate()
model.load_state_dict(torch.load("gate_state_dict.pt", map_location="cpu", weights_only=True))
model.eval()
with torch.no_grad():
    probabilities = model(torch.from_numpy(x)).softmax(dim=1)
```

Provide your trained state dictionary and training-set normalization arrays. Output class order is BASE, RD, DFS, MRP. The HDF5 loader supplies gate features; specialist preprocessing can be connected separately.

## Key ranking

`aes_key_utils.evaluate_ranks(logits, plaintexts, true_keys, encoding="identity")` computes guessing entropy and success rate from your specialist outputs. Supply aligned CPU arrays with logits shaped `[N, 256]` for S-box values, or `[N, 9]` for `encoding="hw"`. Ranks are one-based with conservative ties; success rates are fractions in [0, 1]. Each key is evaluated using its own traces. Repeated permutations do not represent new acquisitions.

## Files

- `gate.py`: four-condition Transformer gate.
- `chameleon_data.py`: Chameleon HDF5 reader and gate feature extraction.
- `aes_constants.py`: AES S-box and Hamming-weight lookup tables.
- `aes_key_utils.py`: candidate scoring and GE/SR evaluation.
- `demo.py`: synthetic interface check (`python demo.py`).
