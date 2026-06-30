import './polyfills' // must precede the tfjs-node import; see polyfills.ts
import * as tf from '@tensorflow/tfjs-node'
import * as nsfw from 'nsfwjs'
import sharp from 'sharp'
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

	static checkSingleImage = async (pic: Buffer, contentType: string) => {

		const model = await NsfwTools.loadModel()

		const image = TFJS_NATIVE.has(contentType)
			? tf.node.decodeImage(pic as Uint8Array, 3) as tf.Tensor3D
			: await NsfwTools.decodeWithSharp(pic)

		const predictions = await model.classify(image)
		image.dispose() // explicit TensorFlow memory management

		return predictions
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

			const predictions = await NsfwTools.checkSingleImage(pic, contentType)

			const topName = predictions[0].className
			const topValue = predictions[0].probability

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

				/* Handle these errors depending on error reason given. */
				const reason: string = e.message.split('\n')[1]

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
}
