"""Pytest session setup.

Caps CPU math-library threads before anything imports numpy/torch, mirroring the
block at the top of scripts/train_mcts.py.  Without it pytest picks up torch's
default intra-op pool (one thread per core -- 32 on framework), whose threads
spin-wait on this project's tiny 7x7 ops.  On a shared box that is ~3000% CPU of
pure contention: it starves a concurrently running training job and makes the
test run itself several times slower.

Must precede the torch import -- the pool is sized at import time, so setting
these afterwards has no effect.  setdefault so an explicit launch-env value wins.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
