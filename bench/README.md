# TF concurrency / batching benchmark

Answers, with numbers, **why nsfw classify throughput is low at low CPU** — before we
commit to any parallelism architecture. We do not assume tfjs-node is single-flight; we
measure it.

## What it measures

For a sweep of concurrency levels `N ∈ {1,2,4,8,16}`:

1. **sequential vs concurrent** — N `classify()` calls one-after-another vs `Promise.all`.
   - `conc/seq ≈ 1` ⇒ concurrent calls **serialize** (single-flight is real).
   - `conc/seq ≫ 1` ⇒ the session overlaps work (a process pool would buy little).
2. **batched vs concurrent** — one raw `model.predict()` on a stacked `[N,299,299,3]` tensor
   vs the concurrent-N above. If batched is much faster **and** fills many cores, **batching
   is the lever** and a child-process pool is likely unnecessary.

CPU utilisation is sampled per variant via `os.cpus()` deltas and reported as **avg busy
cores**. TF threads are left **unbounded** (TF uses all cores) so we see raw capability.

> This plugin has no concept of videos/frames. The sweep is a generic library-level
> concurrency range — deliberately not the host-side "15" (VIDEO_CONCURRENCY × FRAME_BATCH_SIZE).

## Running

Requires **Node 24** (native TypeScript execution — no build step, no tsx). Run from a
checkout of this repo so `@tensorflow/tfjs-node` and `nsfwjs` resolve from `node_modules`
(the exact versions prod runs: tfjs-node 4.22.0, nsfwjs 4.3.0), and the model dir is found
automatically:

```bash
node bench/tf-concurrency.ts
```

Optional flags:

The model dir (`../src/model`, the prod model) and TF deps both resolve from the checkout, so
there are no arguments to pass. One optional flag:

```bash
# load the model + one classify, then exit — smoke-test the environment before the full run
node bench/tf-concurrency.ts --warmup-only
```

To mirror how prod pins threads (AddonComponent sets INTRAOP/INTEROP=8), run it a second time
and compare against the unbounded run above — that delta is likely the whole story:

```bash
TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=8 node bench/tf-concurrency.ts
```

## Reading the result

The final table prints `conc/seq speedup`, `batch/conc speedup`, and `batch busy_cores`.

| observation | conclusion | next step |
|---|---|---|
| `conc/seq ≈ 1` and batch fills few cores | single-flight confirmed | child-process pool |
| `batch ≫ conc` and batch fills many cores | batching is the lever | batch in the plugin (cheap) |
| decode is a large fraction of per-frame time | decode is a bottleneck | offload decode |

Absolute "busy cores" numbers are capped by the host's core count; on a small dev machine
the *ratios* still transfer, but run on the 64-core host for the authoritative answer.
