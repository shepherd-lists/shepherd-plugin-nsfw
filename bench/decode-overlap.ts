/**
 * Decode-overlap benchmark  (run: `npx tsx bench/decode-overlap.ts`)
 *
 * Question: in the real classify path a batch cycle is
 *     decode+preprocess 32 frames  ->  predict(32)  ->  array()
 * all SERIALIZED on the one Node main thread. Prod logs show ~1.4s/batch of which
 * ~1.0s is predict and ~0.4s is on-thread decode that can't overlap predict.
 *
 * Would moving decode OFF the main thread (sharp -> libvips pool) recover that ~0.4s,
 * or does decode then just contend with oneDNN for the same physical cores and gain
 * nothing? Pure-predict benches can't answer this — it depends on core contention, so
 * it MUST be measured on the target box (prod).
 *
 * Three modes, timed per 32-frame batch (best-of-REPEATS wall time, cores sampled):
 *   1. predict-only : predict(32) on a pre-stacked tensor      (floor / pure TF)
 *   2. serial       : tf.node.decodeImage+preprocess 32 frames THEN predict(32)
 *                     (models TODAY's prod: decode serialized on the predict thread)
 *   3. overlap      : kick off sharp decode+resize of the NEXT 32 frames concurrently
 *                     WHILE predict(32) runs, then await both
 *                     (models the proposed fix, WITH real core contention)
 *
 * Read:
 *   overlap ≈ predict-only  => decode fully hid behind predict => the change is worth it
 *   overlap ≈ serial        => contention ate the win          => NOT worth building
 *   gain per batch = serial - overlap  (ms), and the throughput delta it implies
 *
 * SAFE: standalone process. Loads the model and hammers predict/decode; touches nothing
 * in the running classifier, its queues, or the pipeline. Fine to run on prod hardware.
 *
 * Frames: real PNGs written to a temp dir once (PNG entropy-decode is the expensive part;
 * a synthetic in-memory tensor would skip it and understate decode cost).
 */

// --- polyfill: MUST run before @tensorflow/tfjs-node (see src/polyfills.ts) ---
import util from 'node:util'
const u = util as unknown as Record<string, unknown>
if (!u.isNullOrUndefined) u.isNullOrUndefined = (v: unknown): v is null | undefined => v === null || v === undefined
if (!u.isArray) u.isArray = Array.isArray
// -----------------------------------------------------------------------------

import * as tf from '@tensorflow/tfjs-node'
import * as nsfw from 'nsfwjs'
import sharp from 'sharp'
import * as os from 'node:os'
import * as fs from 'node:fs/promises'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
tf.enableProdMode()

const IMG = 299
const REPEATS = 5 // best (min) wall time to cut noise
const modelArg = `file://${path.resolve(__dirname, '..', 'src', 'model')}/`

// ---- args: batch size + optional realistic source frame size ----
const rawArgs = process.argv.slice(2)
const ints = rawArgs.filter(a => /^\d+$/.test(a)).map(Number)
const BATCH = ints[0] ?? 32
// frames are extracted at roughly video resolution then decoded+resized to 299; default
// to a typical keyframe size so decode cost is realistic. override: `... 32 1280 720`
const SRC_W = ints[1] ?? 640
const SRC_H = ints[2] ?? 360

// ---- CPU sampling ----
type CpuSnap = { idle: number; total: number }
const snapCpu = (): CpuSnap => {
	let idle = 0, total = 0
	for (const c of os.cpus()) { for (const t of Object.values(c.times)) total += t; idle += c.times.idle }
	return { idle, total }
}
const busyCores = (a: CpuSnap, b: CpuSnap): number => {
	const dT = b.total - a.total, dI = b.idle - a.idle
	return dT <= 0 ? 0 : (1 - dI / dT) * os.cpus().length
}

type Timed = { ms: number; busy: number }
const timeIt = async (fn: () => Promise<void>, repeats = REPEATS): Promise<Timed> => {
	let best: Timed | null = null
	for (let i = 0; i < repeats; i++) {
		const c0 = snapCpu(); const t0 = performance.now()
		await fn()
		const ms = performance.now() - t0
		const busy = busyCores(c0, snapCpu())
		if (!best || ms < best.ms) best = { ms, busy }
	}
	return best!
}

const rawModel = (m: nsfw.NSFWJS) => (m as unknown as { model: tf.LayersModel }).model
const fmt = (n: number, d = 1) => n.toFixed(d)

// preprocess exactly as NsfwTools does: toFloat -> /255 -> resize if needed. Consumes `decoded`.
const preprocess = (decoded: tf.Tensor3D): tf.Tensor3D =>
	tf.tidy(() => {
		const norm = decoded.toFloat().div(255) as tf.Tensor3D
		const res = (decoded.shape[0] !== IMG || decoded.shape[1] !== IMG)
			? tf.image.resizeBilinear(norm, [IMG, IMG], true)
			: norm
		decoded.dispose()
		return res
	})

// TODAY's path: read + tf.node.decodeImage (main-thread, sync) + preprocess -> [IMG,IMG,3]
const decodeTf = async (file: string): Promise<tf.Tensor3D> => {
	const buf = await fs.readFile(file)
	return preprocess(tf.node.decodeImage(buf as Uint8Array, 3) as tf.Tensor3D)
}

// PROPOSED path: sharp decode+resize off-thread -> raw 299x299x3, then tensor3d+norm (tiny, on-thread)
const decodeSharp = async (file: string): Promise<tf.Tensor3D> => {
	const buf = await fs.readFile(file)
	const { data } = await sharp(buf).resize(IMG, IMG, { fit: 'fill' }).removeAlpha().raw().toBuffer({ resolveWithObject: true })
	return tf.tidy(() => tf.tensor3d(new Uint8Array(data), [IMG, IMG, 3], 'int32').toFloat().div(255) as tf.Tensor3D)
}

// predict over a pre-stacked tensor, disposing only the result (inputs kept for reuse)
const predictStacked = async (stacked: tf.Tensor4D, net: tf.LayersModel) => {
	const rows = tf.tidy(() => net.predict(stacked) as tf.Tensor2D)
	await rows.array()
	rows.dispose()
}

// predict over per-frame tensors (stacks internally, disposes them) — the pipeline shape
const predictBatch = async (imgs: tf.Tensor3D[], net: tf.LayersModel) => {
	const stacked = tf.stack(imgs) as tf.Tensor4D
	await predictStacked(stacked, net)
	stacked.dispose()
	imgs.forEach(t => t.dispose())
}

async function main() {
	console.log('# decode-overlap benchmark')
	console.log(`node=${process.version} cores=${os.cpus().length} backend=${tf.getBackend()} sharp.concurrency=${sharp.concurrency()}`)
	console.log(`TF_NUM_INTRAOP_THREADS=${process.env.TF_NUM_INTRAOP_THREADS ?? '(unset)'} BATCH=${BATCH} src=${SRC_W}x${SRC_H}\n`)

	// write BATCH*2 real PNG frames to a temp dir (need two batches: one to predict, one to decode-ahead)
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'nsfw-overlap-'))
	const files: string[] = []
	for (let i = 0; i < BATCH * 2; i++) {
		const f = path.join(dir, `frame-${i}.png`)
		await sharp({ create: { width: SRC_W, height: SRC_H, channels: 3, background: { r: (i * 37) % 256, g: (i * 91) % 256, b: (i * 53) % 256 } } })
			.png().toFile(f)
		files.push(f)
	}
	const batchA = files.slice(0, BATCH)
	const batchB = files.slice(BATCH, BATCH * 2)

	const model = await nsfw.load(modelArg as `file://${string}`, { size: IMG })
	const net = rawModel(model)
	// warm up graph + threadpools
	await predictBatch(await Promise.all(batchA.map(decodeTf)), net)
	console.log('model loaded + warm.\n')

	// 1. predict-only: decode + stack ONCE outside the timer; time only predict() on the
	//    pre-stacked tensor across all repeats (pure TF, the floor).
	const preImgs = await Promise.all(batchA.map(decodeSharp))
	const preStacked = tf.stack(preImgs) as tf.Tensor4D
	const predictOnly = await timeIt(async () => { await predictStacked(preStacked, net) })
	preStacked.dispose()
	preImgs.forEach(t => t.dispose())

	// 2. serial: decode(32 via tf, on-thread) THEN predict(32) — today's prod cycle
	const serial = await timeIt(async () => {
		const imgs: tf.Tensor3D[] = []
		for (const f of batchA) imgs.push(await decodeTf(f)) // sequential on-thread decode
		await predictBatch(imgs, net)
	})

	// 3. overlap: predict(32) of batch A while sharp-decoding batch B concurrently, await both
	const overlap = await timeIt(async () => {
		const imgsA = await Promise.all(batchA.map(decodeSharp)) // this batch already staged
		const nextDecode = Promise.all(batchB.map(decodeSharp))  // decode-ahead, off-thread
		await predictBatch(imgsA, net)                            // predict runs concurrently with sharp
		const imgsB = await nextDecode
		imgsB.forEach(t => t.dispose())
	})

	const line = (name: string, t: Timed) =>
		`${name.padEnd(12)} | ${fmt(t.ms).padStart(8)} ms | ${fmt(t.ms / BATCH).padStart(6)} ms/frame | busy ${fmt(t.busy)}`
	console.log('mode         |  per-batch |  per-frame | cores')
	console.log('-------------|------------|------------|------')
	console.log(line('predict-only', predictOnly))
	console.log(line('serial', serial))
	console.log(line('overlap', overlap))
	console.log()
	const gain = serial.ms - overlap.ms
	console.log(`serial -> overlap gain: ${fmt(gain)} ms/batch  (${fmt(100 * gain / serial.ms)}%)`)
	console.log(`  throughput: serial ${fmt(BATCH / serial.ms * 1000)} f/s  ->  overlap ${fmt(BATCH / overlap.ms * 1000)} f/s`)
	console.log(`  overlap vs predict-only: ${fmt(overlap.ms - predictOnly.ms)} ms/batch residual`)
	console.log('\nread: overlap≈predict-only => decode hides behind predict, BUILD IT.')
	console.log('      overlap≈serial       => core contention ate it, SKIP IT.')

	await fs.rm(dir, { recursive: true, force: true })
}

main().catch(err => { console.error('benchmark failed:', err); process.exit(1) })
