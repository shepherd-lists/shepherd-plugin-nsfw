import * as tf from '@tensorflow/tfjs-node'
import * as nsfw from 'nsfwjs'
import { logger } from './utils/logger'
import { FilterErrorResult, FilterResult } from 'shepherd-plugin-interfaces'
import si from 'systeminformation'

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const prefix = 'nsfwjs-plugin'

// do this for all envs
tf.enableProdMode()

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

	static checkSingleImage = async (pic: Buffer) => {

		const model = await NsfwTools.loadModel()

		const image = tf.node.decodeImage(pic as Uint8Array, 3) as tf.Tensor3D

		const predictions = await model.classify(image)
		image.dispose() // explicit TensorFlow memory management

		return predictions
	}

	static checkGif = async (gif: Buffer, txid: string): Promise<FilterResult | FilterErrorResult> => {

		const result: FilterResult = {
			flagged: false,
		}

		try {

			const model = await NsfwTools.loadModel()
			const framePredictions = await model.classifyGif(gif, {
				topk: 1,
				fps: 1,
			})

			for (const frame of framePredictions) {
				const class1 = frame[0].className
				const prob1 = frame[0].probability
				result.top_score_name = class1
				result.top_score_value = prob1

				if (['Hentai', 'Porn', 'Sexy'].includes(class1) && prob1 >= 0.9) {
					logger(prefix, `${class1} gif detected`, txid)
					result.flagged = true
					break;
				}
			}

			if (process.env.NODE_ENV === 'test' && !result.flagged) {
				logger(prefix, 'gif clean', txid)
			}

			if (['Neutral', 'Drawing'].includes(result.top_score_name)) {
				result.top_score_name = undefined
				result.top_score_value = undefined
			}


			if (result.top_score_name === 'Porn' && NsfwTools.FALSE_POSITIVE_PORN_SCORES.has(result.top_score_value)) {
				result.flagged = false
				result.top_score_name = undefined
				result.top_score_value = undefined
				logger(prefix, 'false positive porn score detected', txid)
			}


			return result;

		} catch (e) {

			/* handle all the bad data */

			if (
				e.message
				&& (
					e.message === 'Invalid GIF 87a/89a header.'
					|| e.message.startsWith('Unknown gif block:')
					|| e.message.startsWith('Invalid typed array length:')
					|| e.message === 'Invalid block size'
					|| e.message === 'Frame index out of range.'
				)
			) {
				// still not guaranteed to be corrupt, browser may be able to open these
				logger(prefix, `gif. probable corrupt data found (${e.message})`, txid)
				return {
					flagged: undefined,
					data_reason: 'corrupt-maybe',
					err_message: e.message,
				}
			}

			else {
				logger(prefix, 'UNHANDLED error processing gif', txid + ' ', e.name, ':', e.message)
				logger(prefix, await si.mem())
				throw e
			}
		}
	}

	static checkImage = async (pic: Buffer, contentType: string, txid: string): Promise<FilterResult | FilterErrorResult> => {

		// Currently we only support these types:
		if (!["image/bmp", "image/jpeg", "image/png", "image/gif"].includes(contentType)) {
			return {
				flagged: undefined,
				data_reason: 'unsupported',
			}
		}

		// Separate handling for GIFs
		if (contentType === 'image/gif') return NsfwTools.checkGif(pic, txid)

		try {

			const predictions = await NsfwTools.checkSingleImage(pic)

			const topName = predictions[0].className
			const topValue = predictions[0].probability
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

		} catch (e) {

			/* catch all sorts of bad data */

			if (
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
