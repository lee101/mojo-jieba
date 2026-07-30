# mojo-jieba

`mojo-jieba` is a Mojo-accelerated port of the compute-heavy core of
[jieba](https://github.com/fxsjy/jieba), the widely used Chinese word
segmenter. It keeps jieba's familiar Python API while moving dictionary lookup
and maximum-probability route selection into a compiled Mojo shared library.

This is an independent Python module:

```python
import mojo_jieba as jieba
```

The upstream `jieba` 0.42.1 package is currently a runtime dependency for its
MIT-licensed default dictionary and BMES probability tables. Its segmentation
implementation is not called. Keeping the linguistic model identical makes
behavioral parity measurable and avoids silently shipping a different
segmenter under the same API.

## Coverage

The covered API includes:

- accurate segmentation with `cut` and `lcut`, with or without HMM fallback;
- full mode through `cut(..., cut_all=True)`;
- `cut_for_search`, `lcut_for_search`, and `tokenize` with exact offsets;
- `Tokenizer`, custom dictionaries, `set_dictionary`, and `load_userdict`;
- `add_word`, `del_word`, `suggest_freq`, `get_DAG`, and `calc`;
- byte input behavior, whitespace handling, and mixed Chinese/ASCII text.

The Mojo kernel handles the prefix-dictionary candidate search and
backward dynamic program used by accurate and search modes. Jieba's small BMES
unknown-word fallback remains in Python because sparse emission-table lookup
dominates that path.

Not covered are part-of-speech tagging (`jieba.posseg`), keyword extraction
(`jieba.analyse`), Paddle mode, multiprocessing helpers, and command-line
tools. `enable_parallel` raises `NotImplementedError` instead of silently
changing behavior.

## Install and run

The supported development setup uses Pixi:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-jieba.so`. A minimal example:

```bash
pixi run python - <<'PY'
import mojo_jieba as jieba

text = "小明硕士毕业于中国科学院计算所"
print(jieba.lcut(text))
print(list(jieba.tokenize(text, mode="search")))
PY
```

The first call loads jieba's dictionary and constructs the compact native
vocabulary index. Later calls reuse it. Dictionary modifications invalidate
and lazily rebuild that index.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Each row uses identical text and checks that both
implementations return identical results. Times are the median of five warm
runs; speedup is upstream jieba time divided by mojo-jieba time.

| case | mojo-jieba | jieba 0.42.1 | speedup |
|---|---:|---:|---:|
| accurate, HMM=False (196,000 chars) | 115.93 ms | 431.90 ms | 3.73x |
| accurate, HMM=True (196,000 chars) | 282.93 ms | 596.46 ms | 2.11x |
| search, HMM=False (196,000 chars) | 156.49 ms | 902.99 ms | 5.77x |
| tokenize search (196,000 chars) | 228.88 ms | 539.28 ms | 2.36x |

These results include Python regex splitting and result construction on both
sides. The native route kernel is called once per input instead of once per
regex block. Sufficiently large inputs route independent blocks across a
bounded CPU worker pool; smaller inputs stay serial.

No GPU path is provided; this release targets the CPU.

## How it works

Initialization converts jieba's prefix dictionary to a compact
structure-of-arrays trie. Sorted UTF-32 child edges and `int64` child IDs are
grouped by an `int64` node-offset array. A BMP-sized root lookup gives common
characters a direct first-edge lookup, while the small child groups below the
root use short linear scans. A contiguous `float64` array stores terminal
log-probabilities. Those probabilities are computed with Python's `math.log`
once so numerical ties exactly match upstream jieba.

Python encodes an input once and passes its block spans and buffer addresses
through `ctypes`. Before the call, the binding checks every NumPy buffer's
length, dimensionality, dtype, contiguity, alignment, writability, and
non-null address. The C-ABI export reconstructs Mojo pointers using
`AnyOrigin[mut=True]`; no Python objects or strings cross the ABI. Mojo probes
dictionary candidates up to the model's maximum word length, using SIMD for
medium child groups with a scalar remainder, and runs an independent backward
dynamic program for each block into caller-owned `int64` route and `float64`
score arrays. Python then yields substrings with jieba-compatible generator
and offset semantics.

All native memory is owned by NumPy on the Python side. The shared library
does not allocate, retain pointers, or require a matching deallocator.
