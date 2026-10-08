"""Read Chameleon HDF5 operations and generate 1,000 gate features."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

CLASSES = ["BASE", "RD", "DFS", "MRP"]


def load_operations(path, condition, trace_id, limit=100, byte_index=0,
                    sampling_rate=125_000_000):
    """Return unnormalized gate features and aligned evaluation metadata.

    Skip first/last boundary events and require 2,000-sample filter guards.
    Key metadata is returned for labels/evaluation, never as gate input.
    """
    if condition not in CLASSES or not 0 <= byte_index < 16 or limit < 1:
        raise ValueError("Invalid condition, byte index, or limit")
    if sampling_rate <= 250_000:
        raise ValueError("Sampling rate must exceed twice the 125 kHz cutoff")
    sos = butter(3, 125_000, btype="highpass", fs=sampling_rate, output="sos")
    with h5py.File(path, "r") as f:
        meta = f[f"metadata/ciphers/ciphers_{trace_id}"]
        plain = meta["plaintexts"][:]["p"]
        key = meta["key"][0]["k"]
        pins = f[f"metadata/pinpoints/pinpoints_{trace_id}"][:]
        wave = f[f"data/traces/trace_{trace_id}"]
        if len(plain) != len(pins):
            raise ValueError("Plaintext and pinpoint counts differ")
        features, events = [], []
        for event in range(1, len(pins) - 1):
            start, end = int(pins[event]["start"]), int(pins[event]["end"])
            if start < 2000 or end-start < 100000 or start+102000 > len(wave):
                continue
            raw = np.asarray(wave[start-2000:start+102000], dtype=np.float32)
            filtered = sosfiltfilt(sos, raw)[2000:102000]
            features.append(filtered.reshape(1000, 100).mean(1))
            events.append(event)
            if len(events) == limit:
                break
        if not events:
            raise ValueError("No eligible AES operations in this trace")
        return dict(x=np.asarray(features, dtype=np.float32),
                    condition=np.full(len(events), CLASSES.index(condition), dtype=np.int64),
                    plaintexts=np.asarray(plain[events, byte_index], dtype=np.uint8),
                    true_keys=np.full(len(events), key[byte_index], dtype=np.uint8),
                    event=np.asarray(events, dtype=np.int64),
                    trace_id=np.full(len(events), trace_id, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=CLASSES)
    parser.add_argument("--trace-id", required=True, type=int)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--byte-index", default=0, type=int)
    parser.add_argument("--sampling-rate", default=125_000_000, type=float)
    parser.add_argument("--output", default=Path("features.npz"), type=Path)
    args = parser.parse_args()
    data = load_operations(args.h5, args.condition, args.trace_id,
                           args.limit, args.byte_index, args.sampling_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Saved {len(data['x'])} operations to {args.output}")


if __name__ == "__main__":
    main()
