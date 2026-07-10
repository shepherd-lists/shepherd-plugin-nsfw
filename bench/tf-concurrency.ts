/**
 * TF concurrency / batching benchmark  (run: `node bench/tf-concurrency.ts`)
 *
 * Purpose: settle, with numbers, WHY nsfw classify throughput is low at low CPU.
 * We do NOT assume tfjs-node is single-flight — we measure it. Three questions:
 *
 *   1. Do N concurrent classify() calls on one session serialize?
 *        (sequential-N total time  vs  Promise.all(N) time)
 *        concurrent ≈ sequential  => single-flight is real  => a process pool is justified
 *        concurrent ≪ sequential  => the session overlaps    => pool buys little
 *
 *   2. Does one batched [N,299,299,3] classify beat N concurrent single calls,
 *      and does it fill more CPU cores?   (the "does batching alone win?" test)
 *        batched much faster + fills many cores => batching is the lever, pool may be moot
 *
 * Config: TF threads left UNBOUNDED (TF picks core count). CPU utilisation sampled
 * per variant via os.cpus() deltas => "avg busy cores".
 *
 * This plugin has no concept of videos/frames — the sweep is a generic library-level
 * concurrency range, deliberately not the host-side "15".
 */

// --- polyfill: MUST run before @tensorflow/tfjs-node (see src/polyfills.ts) ---
// tfjs-node imports util.isArray / util.isNullOrUndefined, removed in Node 22+.
import util from 'node:util'
const u = util as unknown as Record<string, unknown>
if (!u.isNullOrUndefined) {
	u.isNullOrUndefined = (val: unknown): val is null | undefined => val === null || val === undefined
}
if (!u.isArray) {
	u.isArray = Array.isArray
}
// -----------------------------------------------------------------------------

import * as tf from '@tensorflow/tfjs-node'
import * as nsfw from 'nsfwjs'
import * as os from 'node:os'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

tf.enableProdMode()

const IMG = 299 // nsfwjs model input size (matches NsfwTools loadModel { size: 299 })
const REPEATS = 3 // repeat each variant, report best (min) wall time to cut noise

// ---- args ----
const rawArgs = process.argv.slice(2)
const args = new Set(rawArgs.filter(a => a.startsWith('--')).map(a => a.slice(2)))
// --warmup: load model + one classify, then exit (smoke-test the env)
const warmupOnly = args.has('warmup')
// --batch-only: skip the flat sequential/concurrent baselines and run just the batched
// sweep. The derived conc/seq table is replaced by a batched thru/busy_cores summary.
// Use it to quickly find the batch-size knee or sweep TF thread counts.
const batchOnly = args.has('batch-only')
// positional ints override the sweep, e.g. `--batch-only 16 32 48 64 96`
const customSweep = rawArgs.filter(a => /^\d+$/.test(a)).map(Number)
const CONCURRENCY_SWEEP = customSweep.length ? customSweep : [1, 2, 4, 8, 16, 32, 64] // generic; NOT the host-side video number

// the repo's own model dir (== prod model)
const modelArg = `file://${path.resolve(__dirname, '..', 'src', 'model')}/`

// ---- CPU sampling (os.cpus deltas -> average busy cores over an interval) ----
type CpuSnap = { idle: number; total: number }
const snapCpu = (): CpuSnap => {
	let idle = 0, total = 0
	for (const c of os.cpus()) {
		for (const t of Object.values(c.times)) total += t
		idle += c.times.idle
	}
	return { idle, total }
}
/** average number of fully-busy cores between two snapshots */
const busyCores = (a: CpuSnap, b: CpuSnap): number => {
	const dTotal = b.total - a.total
	const dIdle = b.idle - a.idle
	if (dTotal <= 0) return 0
	const busyFraction = 1 - dIdle / dTotal
	return busyFraction * os.cpus().length
}

// ---- timing helper: runs `fn`, samples CPU across it, returns best-of-REPEATS ----
type Timed = { ms: number; busy: number }
const timeIt = async (fn: () => Promise<void>, repeats = REPEATS): Promise<Timed> => {
	let best: Timed | null = null
	for (let i = 0; i < repeats; i++) {
		const c0 = snapCpu()
		const t0 = performance.now()
		await fn()
		const ms = performance.now() - t0
		const busy = busyCores(c0, snapCpu())
		if (!best || ms < best.ms) best = { ms, busy }
	}
	return best!
}

// ---- synthetic input: random uint8 image data as an int32 Tensor3D [299,299,3] ----
// int32 mirrors what NsfwTools feeds classify() (decodeImage/sharp both yield int32 pixels).
const makeFrame = (): tf.Tensor3D =>
	tf.tidy(() => tf.randomUniform([IMG, IMG, 3], 0, 256, 'int32') as tf.Tensor3D)

/**
 * Build the batched input the way nsfwjs.infer() would, but for N images at once:
 * toFloat -> /255 -> already 299x299 (no resize) -> [N,299,299,3]. The underlying
 * tf model.predict() accepts any batch dim; nsfwjs.classify() cannot (it hardcodes
 * reshape([1,size,size,3])), which is exactly why the batched test goes to the raw model.
 */
const makeBatch = (n: number): tf.Tensor4D =>
	tf.tidy(() =>
		tf.stack(Array.from({ length: n }, () =>
			tf.randomUniform([IMG, IMG, 3], 0, 256, 'int32').toFloat().div(255) as tf.Tensor3D,
		)) as tf.Tensor4D,
	)

// nsfwjs stores the tf model at `.model`; predict() is the raw batched forward pass.
const rawModel = (m: nsfw.NSFWJS) => (m as unknown as { model: tf.LayersModel | tf.GraphModel }).model

const fmt = (n: number, d = 1) => n.toFixed(d)

async function main() {
	console.log('# TF concurrency / batching benchmark')
	console.log(`node=${process.version} cores=${os.cpus().length} backend=${tf.getBackend()}`)
	console.log(`TF_NUM_INTRAOP_THREADS=${process.env.TF_NUM_INTRAOP_THREADS ?? '(unset/unbounded)'}`)
	console.log(`TF_NUM_INTEROP_THREADS=${process.env.TF_NUM_INTEROP_THREADS ?? '(unset/unbounded)'}`)
	console.log(`model=${modelArg}`)

	const model = await nsfw.load(modelArg as `file://${string}`, { size: IMG })
	console.log('model loaded.\n')

	// warm up the graph/threadpool so the first timed run isn't skewed
	{
		const f = makeFrame()
		await model.classify(f)
		f.dispose()
	}
	if (warmupOnly) { console.log('warmup-only: done.'); return }

	// -------------------------------------------------------------------------
	// Measurements 1 & 2, per concurrency N
	// -------------------------------------------------------------------------
	console.log('variant     | N  | wall_ms | thru(clf/s) | busy_cores')
	console.log('------------|----|---------|-------------|-----------')

	type Row = { seq: Timed | null; conc: Timed | null; batch: Timed }
	const rows = new Map<number, Row>()

	const net = rawModel(model)

	for (const N of CONCURRENCY_SWEEP) {
		// Inputs are built fresh INSIDE each timed closure so every repeat is self-contained
		// (no use-after-dispose) and all three variants pay the same cheap randomUniform cost.
		// That construction cost is negligible next to a model forward pass.

		// (1a) SEQUENTIAL: N classify() calls one after another
		const seq = batchOnly ? null : await timeIt(async () => {
			for (let i = 0; i < N; i++) {
				const f = makeFrame()
				await model.classify(f)
				f.dispose()
			}
		})

		// (1b) CONCURRENT: same N as Promise.all
		const conc = batchOnly ? null : await timeIt(async () => {
			const frames = Array.from({ length: N }, makeFrame)
			await Promise.all(frames.map(f => model.classify(f)))
			frames.forEach(f => f.dispose())
		})

		// (2) BATCHED: one raw model.predict() on a stacked [N,299,299,3] tensor.
		//     Goes to the underlying tf model because nsfwjs.classify() can't batch
		//     (it hardcodes reshape([1,...])). Preprocessing (float, /255) matches infer().
		const batch = await timeIt(async () => {
			const batched = makeBatch(N)
			const logits = net.predict(batched) as tf.Tensor
			await logits.data() // force the async compute to actually complete before we stop the clock
			logits.dispose()
			batched.dispose()
		})

		rows.set(N, { seq, conc, batch })

		const line = (name: string, t: Timed) =>
			`${name.padEnd(11)} | ${String(N).padStart(2)} | ${fmt(t.ms).padStart(7)} | ${fmt((N / t.ms) * 1000, 1).padStart(11)} | ${fmt(t.busy)}`
		if (seq) console.log(line('sequential', seq))
		if (conc) console.log(line('concurrent', conc))
		console.log(line('batched', batch))
		console.log('------------|----|---------|-------------|-----------')
	}

	// -------------------------------------------------------------------------
	// Derived signals: the actual decision. In --batch-only mode there are no
	// seq/conc baselines, so we just chart batched throughput + core fill.
	// -------------------------------------------------------------------------
	if (batchOnly) {
		console.log('\n# batched sweep — watch thru + busy_cores climb, then plateau (the knee)')
		console.log('N  | thru(clf/s) | busy_cores')
		console.log('---|-------------|-----------')
		for (const N of CONCURRENCY_SWEEP) {
			const r = rows.get(N)!
			console.log(`${String(N).padStart(2)} | ${fmt((N / r.batch.ms) * 1000, 1).padStart(11)} | ${fmt(r.batch.busy).padStart(10)}`)
		}
		console.log(
			'\nread: thru still rising + busy_cores climbing => under-batched, raise NSFW_BATCH_SIZE.' +
			'\n      thru/busy_cores flatten => single predict is maxed on this silicon (physical-core bound).',
		)
	} else {
		console.log('\n# derived signals (higher speedup = better)')
		console.log('N  | conc/seq speedup | batch/conc speedup | batch busy_cores')
		console.log('---|------------------|--------------------|-----------------')
		for (const N of CONCURRENCY_SWEEP) {
			const r = rows.get(N)!
			const concSpeedup = r.seq!.ms / r.conc!.ms   // ~1 => concurrency serializes (single-flight)
			const batchSpeedup = r.conc!.ms / r.batch.ms // >1 => batching wins over concurrency
			console.log(
				`${String(N).padStart(2)} | ${fmt(concSpeedup, 2).padStart(16)} | ${fmt(batchSpeedup, 2).padStart(18)} | ${fmt(r.batch.busy).padStart(16)}`,
			)
		}
		console.log(
			'\nread: conc/seq≈1 AND batch fills few cores => single-flight confirmed (pool justified).' +
			'\n      batch≫conc AND batch fills many cores => batching is the lever (pool likely moot).',
		)
	}
}

main().catch(err => {
	console.error('benchmark failed:', err)
	process.exit(1)
})
