"""Product-quantize GlotLID so it fits in about 350 MB of RAM instead of 1.8 GB.

Needs the full fastText build, which conflicts with fasttext-predict (same module name),
so run it in an isolated environment rather than the project's:

    uv run --no-project --python 3.12 --with fasttext-wheel==0.9.2 --with numpy==1.26.4 \
        python scripts/compress_glotlid.py models/glotlid/model.bin models/glotlid-q.ftz

fasttext-wheel has no macOS arm64 wheel for Python 3.12; on a Mac, run this in Docker or on
another machine and copy the output file.
"""

import sys
import time
from pathlib import Path

import fasttext


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.time()
    model = fasttext.load_model(str(source))
    print(f"Loaded {source} ({source.stat().st_size / 1e6:.0f} MB) in {time.time() - started:.1f} s")

    started = time.time()
    # cutoff (vocabulary pruning) needs the original training data, so only the matrices are quantized.
    model.quantize(input=None, retrain=False, qnorm=True, qout=False, cutoff=0, dsub=2)
    model.save_model(str(target))
    ratio = 100 * target.stat().st_size / source.stat().st_size
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB, {ratio:.1f}% of the original) "
          f"in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
