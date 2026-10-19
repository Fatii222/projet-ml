"""Trains (or reloads) the 6 models in artifacts/: useful before a deployment (avoids waiting on first launch)."""
import time

from targets import TARGETS

for t in TARGETS:
    t0 = time.time()
    bundle = t.load()
    for entry in bundle["models"].values():
        print(f"[{t.slug:<12}] {entry['name']:<24} {entry['metrics']}  ({time.time() - t0:.0f}s)")
