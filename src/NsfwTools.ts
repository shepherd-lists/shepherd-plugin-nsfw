import './polyfills' // must precede the tfjs-node import; see polyfills.ts
import * as tf from '@tensorflow/tfjs-node'
import * as nsfw from 'nsfwjs'
import sharp from 'sharp'
import fs from 'fs/promises'
import path from 'path'
import { logger } from './utils/logger'
import { FilterErrorResult, FilterResult } from 'shepherd-plugin-interfaces'
import si from 'systeminformation'

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const prefix = 'nsfwjs-plugin'

// do this for all envs
tf.enableProdMode()

// content types tfjs-node can decode natively (via tf.node.decodeImage).
// anything else is routed through sharp and decoded to raw RGB pixels.
const TFJS_NATIVE = new Set(['image/bmp', 'image/jpeg', 'image/png'])

// nsfwjs class order (matches nsfwjs's NSFW_CLASSES map, index 0..4)
const NSFW_CLASSES = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy'] as const

const IMAGE_SIZE = 299 // model input dimension (matches loadModel { size: 299 })

// Max images per model.predict() call. tfjs-node is single-flight, so every
// classify in the process funnels through ONE shared batcher and is grouped
// into predicts of up to this size. The batch size that saturates a host is
// hardware-dependent, so it is set per-host via env.
const NSFW_BATCH_SIZE = (() => {
	const n = Number(process.env.NSFW_BATCH_SIZE)
	return Number.isFinite(n) && n > 0 ? Math.floor(n) : 16
})()
// How long the batcher waits for more images before flushing a partial batch.
// One predict costs hundreds of ms, so this is negligible; it only bounds
// latency for the trickle case (a lone image / a video's last partial batch).
const BATCH_WAIT_MS = 100 //ms

// one image's top prediction, mirroring nsfwjs predictions[0]
type TopPrediction = { className: string; probability: number }
// a queued classify request: the preprocessed [size,size,3] image + its resolver
type BatchItem = {
	image: tf.Tensor3D
	resolve: (p: TopPrediction) => void
	reject: (e: unknown) => void
}

export class NsfwTools {
	private static _isLoading = false
	private static _model: nsfw.NSFWJS
	private constructor() { } //hide

	private static readonly FALSE_POSITIVE_PORN_SCORES = new Set([
		0.903248131275177,
		0.9651741981506348,
		0.9032477140426636,
		0.9885595440864563,
	])

	static async init() {
		await NsfwTools.loadModel()
	}

	static async loadModel() {
		//wait if model in process of being loaded
		while (this._isLoading) await sleep(100)
		if (NsfwTools._model) {
			// model already loaded
			return NsfwTools._model
		}
		this._isLoading = true
		logger(prefix, 'loading model once')
		// model folder is here also: LN6kloFszCgXvubWNvbRHpp4DCnCLnXQakz8SplJZFQ
		NsfwTools._model = await nsfw.load(`file://${__dirname}/model/`, { size: 299 })
		this._isLoading = false
		return NsfwTools._model
	}

	/* ----------------------------------------------------------------------
	 * Shared dynamic batcher.
	 *
	 * tfjs-node holds ONE native session per process and is single-flight:
	 * concurrent classify calls serialize, and a single-image forward pass
	 * cannot fill the machine. So every classify request in the process -
	 * whether a lone checkImage or a frame from checkImageDir - is submitted
	 * here, grouped into one model.predict() on a stacked [n,size,size,3]
	 * tensor, and its per-row result routed back to the caller. Batching a
	 * stack is what actually saturates the cores (see benchmarks).
	 *
	 * A batch is purely a transport optimization: its rows are unrelated
	 * images (a standalone tx, a frame of video X, a frame of gif Y). Each row
	 * is decoded independently and returned to its own submitter - no
	 * batch-level/cross-row reduction is ever done.
	 * -------------------------------------------------------------------- */
	private static _queue: BatchItem[] = []
	private static _flushTimer: NodeJS.Timeout | null = null

	/** Submit a preprocessed [size,size,3] float image; resolves with its top prediction. */
	static classifyBatched = (image: tf.Tensor3D): Promise<TopPrediction> =>
		new Promise<TopPrediction>((resolve, reject) => {
			NsfwTools._queue.push({ image, resolve, reject })
			// full batch -> flush immediately; otherwise arm the wait timer once
			if (NsfwTools._queue.length >= NSFW_BATCH_SIZE) {
				NsfwTools.flushQueue()
			} else if (!NsfwTools._flushTimer) {
				NsfwTools._flushTimer = setTimeout(NsfwTools.flushQueue, BATCH_WAIT_MS)
			}
		})

	/** Run one predict() over up to NSFW_BATCH_SIZE queued images and resolve each. */
	private static flushQueue = async () => {
		if (NsfwTools._flushTimer) {
			clearTimeout(NsfwTools._flushTimer)
			NsfwTools._flushTimer = null
		}
		if (NsfwTools._queue.length === 0) return //not certain how this could occur

		const batch = NsfwTools._queue.splice(0, NSFW_BATCH_SIZE)
		// more than one batch's worth is waiting -> keep draining after this one
		if (NsfwTools._queue.length > 0 && !NsfwTools._flushTimer) {
			NsfwTools._flushTimer = setTimeout(NsfwTools.flushQueue, BATCH_WAIT_MS)
		}

		try {
			const model = await NsfwTools.loadModel()
			// nsfwjs stores the underlying tf model at .model; predict() takes any
			// batch dim (nsfwjs.classify() cannot - it hardcodes reshape([1,...])).
			const net = (model as unknown as { model: tf.LayersModel }).model

			// [n,size,size,3] top predictions, computed in one forward pass.
			const rows = tf.tidy(() => {
				const stacked = tf.stack(batch.map(b => b.image)) as tf.Tensor4D
				return net.predict(stacked) as tf.Tensor2D
			})
			const scores = await rows.array() // number[][], one row of 5 per image
			rows.dispose()

			// route each row back to its own submitter - strictly per row
			for (let i = 0; i < batch.length; i++) {
				const row = scores[i]
				let top = 0
				for (let c = 1; c < row.length; c++) if (row[c] > row[top]) top = c
				batch[i].resolve({ className: NSFW_CLASSES[top], probability: row[top] })
			}
		} catch (e) {
			batch.forEach(b => b.reject(e))
		} finally {
			batch.forEach(b => b.image.dispose())
		}
	}

	/** Apply the flag decision to one image's top prediction. Shared by both paths. */
	private static topPredictionToResult = (topName: string, topValue: number, txid: string): FilterResult => {
		if (topName === 'Porn' && NsfwTools.FALSE_POSITIVE_PORN_SCORES.has(topValue)) {
			logger(prefix, 'false positive porn score detected', txid)
			return { flagged: false }
		}

		const flagged = (['Sexy', 'Porn', 'Hentai'].includes(topName)) && topValue >= 0.9

		if (flagged) {
			logger(prefix, JSON.stringify({ txid, flagged, topName, topValue }))
		}

		return {
			flagged,
			...(['Porn', 'Sexy', 'Hentai'].includes(topName) && {
				top_score_name: topName,
				top_score_value: topValue,
			})
		}
	}

	/**
	 * Preprocess a decoded image tensor exactly as nsfwjs.infer() does, so batched
	 * scores match classify() bit-for-bit: toFloat -> /255 -> resize to size if needed.
	 * Consumes `decoded` (disposed here).
	 */
	private static preprocess = (decoded: tf.Tensor3D): tf.Tensor3D =>
		tf.tidy(() => {
			const normalized = decoded.toFloat().div(255) as tf.Tensor3D
			const resized = (decoded.shape[0] !== IMAGE_SIZE || decoded.shape[1] !== IMAGE_SIZE)
				? tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], true)
				: normalized
			decoded.dispose()
			return resized
		})

	static checkSingleImage = async (pic: Buffer, contentType: string): Promise<TopPrediction> => {

		await NsfwTools.loadModel()

		const decoded = TFJS_NATIVE.has(contentType)
			? tf.node.decodeImage(pic as Uint8Array, 3) as tf.Tensor3D
			: await NsfwTools.decodeWithSharp(pic)

		// batcher disposes the submitted (preprocessed) tensor after predict
		return NsfwTools.classifyBatched(NsfwTools.preprocess(decoded))
	}

	/**
	 * Decode any sharp-supported format (WebP, AVIF, TIFF, SVG, HEIC, GIF, ...)
	 * into a 3-channel RGB Tensor3D. First frame only for animated inputs.
	 * Throws sharp's own error if the buffer cannot be decoded.
	 */
	static decodeWithSharp = async (pic: Buffer): Promise<tf.Tensor3D> => {
		const { data, info } = await sharp(pic, { animated: false })
			.rotate()                 // honour EXIF orientation
			.toColourspace('srgb')    // grayscale/CMYK/etc. -> 3-channel RGB(+A)
			.removeAlpha()            // drop alpha -> exactly 3 channels
			.raw()
			.toBuffer({ resolveWithObject: true })

		return tf.tensor3d(
			new Uint8Array(data),
			[info.height, info.width, 3],
			'int32',
		)
	}

	static checkImage = async (pic: Buffer, contentType: string, txid: string): Promise<FilterResult | FilterErrorResult> => {

		try {

			const top = await NsfwTools.checkSingleImage(pic, contentType)

			return NsfwTools.topPredictionToResult(top.className, top.probability, txid)

		} catch (err: unknown) {

			/* catch all sorts of bad data */
			const e = err as Error

			if (
				/* sharp could not decode the buffer (format libvips wasn't built
					 with, or data sharp considers undecodable) */
				!TFJS_NATIVE.has(contentType)
				&& /unsupported image format|corrupt header|Input buffer contains|premature end|VipsForeignLoad/i.test(e.message)
			) {
				logger(prefix, 'sharp could not decode image', contentType, txid)
				return {
					flagged: undefined,
					data_reason: 'unsupported',
				}
			}

			else if (
				e.message === 'Expected image (BMP, JPEG, PNG, or GIF), but got unsupported image type'
				&& (['image/bmp', 'image/jpeg', 'image/png'].includes(contentType)) //sanity, should already be checked
			) {
				logger(prefix, 'probable corrupt data found', contentType, txid)
				return {
					flagged: undefined, //undefined as not 100% sure, might be tfjs problem opening file
					data_reason: 'corrupt-maybe',
				}
			}

			else if (e.message.startsWith('Invalid TF_Status: 3')) {

				/* Handle these errors depending on error reason given.
					 The native binding formats these as "Invalid TF_Status: 3\nMessage: ...".
					 Default to '' if that second line is ever missing so the matches below
					 fall through to the UNHANDLED branch instead of throwing a TypeError. */
				const reason: string = e.message.split('\n')[1] ?? ''

				if (
					reason.startsWith('Message: Invalid PNG data, size')
					|| reason === 'Message: jpeg::Uncompress failed. Invalid JPEG data or crop window.'
					|| reason.startsWith('Message: Input size should match (header_size + row_size * abs_height) but they differ by')
				) {
					//partial image
					logger(prefix, 'partial image found', contentType, txid)
					return {
						flagged: undefined,
						data_reason: 'partial',
					}
				}

				else if (reason.startsWith('Message: PNG size too large for int:')) {
					//oversized png
					logger(prefix, 'oversized png found', contentType, txid)
					return {
						flagged: undefined,
						data_reason: 'oversized',
					}
				}

				else if (
					reason.startsWith('Message: Number of channels inherent in the image must be 1, 3 or 4, was')
				) {
					// unreadable data
					// logger(prefix, 'bad data found', contentType, url)
					// await dbCorruptDataConfirmed(txid)
					return {
						flagged: undefined, //error signal, this will be flagged false
						data_reason: 'corrupt',
					}
				}

				else if (reason === 'Message: Invalid PNG. Failed to initialize decoder.') {
					// unknown issue - too big maybe? these images are opening in the browser.
					logger(prefix, 'treating as partial.', e.message, contentType, txid)
					return {
						flagged: undefined,
						data_reason: 'partial',
					}
				}

				else {
					logger(prefix, 'UNHANDLED "Invalid TF_Status: 3" found. Reason:', `"${reason}"`, contentType, txid)
					throw e
				}
			}

			else if (e.message.startsWith('Invalid TF_Status: 8')) {
				// OOM error. handle later.
				logger(prefix, await si.mem())
				return {
					flagged: undefined,
					data_reason: 'oversized',
				}
			}

			else {
				logger(prefix, `UNHANDLED error processing [${txid}]`, e.name, ':', e.message)
				logger(prefix, await si.mem())
				throw e
			}
		}
	}

	/**
	 * Classify a folder of extracted video/gif keyframes (`frame-<n>.png`, produced
	 * by us, so always decodable PNG). Every frame is submitted to the shared batcher
	 * and classified across the whole process alongside any concurrent checkImage /
	 * checkImageDir calls. Frames are read in numeric order; the FIRST flagged frame
	 * wins, else not-flagged. Throws if the folder contains no frames.
	 */
	static checkImageDir = async (framesDir: string, _mimetype: string, txid: string): Promise<FilterResult | FilterErrorResult> => {

		await NsfwTools.loadModel()

		const frames = (await fs.readdir(framesDir))
			.filter(name => /^frame-\d+\.png$/.test(name))
			.sort((a, b) => Number(a.match(/\d+/)![0]) - Number(b.match(/\d+/)![0]))

		if (frames.length === 0) {
			throw new Error(`checkImageDir: no frame-*.png files in ${framesDir} [${txid}]`)
		}

		// submit every frame to the shared batcher; results come back in frame order
		const results = await Promise.all(frames.map(async name => {
			const pic = await fs.readFile(path.join(framesDir, name))
			const decoded = tf.node.decodeImage(pic as Uint8Array, 3) as tf.Tensor3D
			return NsfwTools.classifyBatched(NsfwTools.preprocess(decoded))
		}))

		// first flagged frame wins (frame order preserved by Promise.all)
		for (const top of results) {
			const res = NsfwTools.topPredictionToResult(top.className, top.probability, txid)
			if (res.flagged) return res
		}

		return { flagged: false }
	}
}
