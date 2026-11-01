# Reusable ordinary staging for EXL3 paging experiments

This standalone experiment borrows the useful indirection and capacity tradeoff
from Betelgeuse: logical expert IDs continue to select resident transforms;
physical slot IDs select only trellis records. It changes no engine code, loader,
pinned hash, default path, or production behavior. The CUDA shim includes the
existing native implementation and uses `Experts::run_rows` launch geometry.

The default pager allocates one ordinary host buffer for its eight-record
capacity and reuses it across requests. It validates every missing record before
uploading any payload or publishing the next slot mapping. This avoids fresh
record allocation and zeroing on each miss. The serving engine currently keeps
expert weights resident; this component is a foundation for a future pager,
not a serving-path optimization.

The default fixture uses **synthetic weights**: one rank of one GLM-shaped layer, 288 experts,
4096-wide input, 512-wide rank-local intermediate, eight routed experts and shared
sentinel 288. Each expert has three contiguous 1 MiB gate/up/down trellises.
Six nonuniform deterministic FP16 transform arrays remain resident; inputs are
synthetic BF16. Optional checkpoint-derived weights use the same synthetic input and routes; neither mode is a full-model quality test.
The shared-expert output is a zero-filled sentinel, not a shared-expert model.

## Reproduce on a GB10 host

Needs CUDA 13 at `/usr/local/cuda`, C++ linker tools and Rust (edition 2024).
The experiment uses only Rust's standard library and CUDA runtime. For an
isolated Rust installation, obtain the official installer from
<https://rustup.rs>, set `RUSTUP_HOME` and `CARGO_HOME` under
`~/.cache/betelgeuse-experiment-tools`, and use `--profile minimal --no-modify-path`.
It does not need the full engine's private artifacts or Rust dependencies.

```sh
./experiments/betelgeuse-paging/run.sh \
  "$HOME/.local/state/betelgeuse-experiment/$(date -u +%Y%m%dT%H%M%SZ)"
```

The script keeps generated executables, the ~864 MiB pack, logs and results
outside the source checkout. Each run requires a fresh evidence directory. Each record's
identity, length and standard-library checksum is verified before use. The checksum uses Rust's `DefaultHasher` to detect accidental fixture corruption;
it is not an authentication scheme. This is an internal fresh-run file format,
so source hashes and Rust version identify the algorithm and every run creates
its own pack.
Source/native/binary hashes and tool versions accompany each run.

## What it proves and measures

- An eight-slot synchronous pager protects every member of the current active
  union, including hits that appear after misses. It rejects invalid, duplicate
  or over-capacity active IDs. Every missing record is read and validated before
  the first upload; a failed read never publishes a new mapping.
- Resident, all-hit pool, alternating four-miss overlap, and alternating
  eight-miss eviction tracks run eagerly and through a reusable CUDA graph.
  Each track has five warmups and 200 measured repetitions; percentiles use
  nearest rank. Graph capture happens during warmup.
- Every repetition must match the resident eager reference bit for bit, with all
  outputs finite. A deliberate wrong slot map must change outputs before normal
  measurements proceed. This is exact arithmetic/mapping parity only; it cannot
  establish model quality, perplexity, text quality, or end-to-end serving latency.
- A separate 36-request disjoint track requests every expert once after a
  successful own-file `POSIX_FADV_DONTNEED` advisory. It has no warmups; physical
  `read_bytes` establishes whether storage was actually read. Its p99 is the
  maximum sample, and its first graph replay may include setup. No global cache
  state is changed.
- Host latency includes mapping, synchronous buffered `pread` via `read_exact_at`,
  checksum, staging, CUDA transfers, compute, synchronization, and output readback.
  It excludes output parity comparison and one-time setup/reference construction.
  GPU latency is CUDA-event time around only the five kernels or graph replay.
  Fill latency includes CPU mapping, reads, checksum and synchronous upload.
  Separate median read, validation/checksum, and upload times expose CPU hashing
  cost; their sum need not equal total fill due to staging allocations and mapping.
- Logical payload counts exclude 4096-byte per-record headers; `/proc/self/io`
  `read_bytes` reports physical storage reads. Record and payload boundaries are 4KiB-aligned. The generated pack is ordinarily
  warm in Linux page cache. Zero physical bytes is not NVMe performance evidence.
- Trellis allocations are 864 MiB resident versus 24 MiB pool (36x difference);
  resident transforms total 7,962,624 bytes in either case. The reference allocation
  coexists with the pool in this process, so these are component byte counts,
  not observed process-memory savings. Allocating host staging scales with current misses; reusable modes reserve a fixed capacity.

Graph replay is already an engine feature and is included as a compatibility and
launch-overhead measurement. For optional checkpoint-derived coefficients, pass a second argument to `run.sh`:
an external directory containing `trellises.raw` (288 consecutive `gate||up||down`
records of three 1MiB arrays) and `transforms.raw` (six concatenated FP16 arrays:
`suh_g[288,4096]`, `suh_u[288,4096]`, `suh_d[288,512]`, `svh_g[288,512]`,
`svh_u[288,512]`, `svh_d[288,4096]`). Lengths are checked before GPU use; inputs and
routing remain synthetic. The source extraction/provenance must be supplied
separately. This tests the supplied slice, not conversion or trained-model quality.
Both paths generate a checksum-bearing aligned record pack outside Git.

No async GPU-to-host doorbell, producer/consumer
protocol, overlapping I/O, route prediction, cross-layer eviction policy, production
cache integration, or model quality benchmark is implemented here.

## Reusable host staging comparison

The optional third positional argument selects `reused` (default), `allocating`,
or `pinned`; e.g. `probe OUT MODEL_SLICE_DIR allocating`. Use `-` as the model directory
for synthetic coefficients. `run.sh` forwards these same optional arguments.

Allocating and pinned modes are benchmark controls. Ordinary reuse is the
recommended path; pinning did not consistently improve the measured copies.

`allocating` retains fresh per-miss record vectors. `reused` keeps one ordinary
`Vec` sized for eight full records. `pinned` keeps one equally sized
`cudaHostAlloc` allocation until synchronous uploads finish. Fixed staging is
25,198,592 host bytes including headers; allocation and initialization are outside
request timing. Allocating mode releases payload buffers at each successful fill.
Every mode reads and validates all missing records before the first upload and
uses identical records, routes, native arithmetic, and CUDA-copy calls. Reuse and
pinning are separate components; upload happens outside graph replay. Compare
forward and reverse execution orders before attributing a latency change to the
chosen staging component.

## Prepare checkpoint-derived coefficients

The offline extractor requires only Python's standard library. Supply the cached
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` snapshot directory and a fresh output
directory outside this checkout:

```sh
python3 experiments/betelgeuse-paging/extract_layer.py --self-test
python3 experiments/betelgeuse-paging/extract_layer.py \
  --source "$SOURCE_SNAPSHOT" --output "$SLICE_DIR"
./experiments/betelgeuse-paging/run.sh "$RUN_DIR" "$SLICE_DIR" reused
```

It extracts layer 3, rank 0 of a proposed TP4 layout, validates tensor metadata,
source extents, MCG constants and finite transforms, and publishes a provenance
manifest last. Wait for that manifest and verify its output hashes before reuse.
The cached revision tested was `25a44fdbf16862a46b7cc9921142c6c81350af2f`.

This is an unqualified slice extractor, not the engine's missing prepared-model
converter. A parity pass establishes that the pager preserves the supplied
coefficients' outputs, not that the TP4 slicing or full model is correct.
