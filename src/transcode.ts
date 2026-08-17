import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { logger } from './utils/logger'

const prefix = 'nsfw-plugin'

/**
 * Hard wall-clock limit on one ffmpeg run. In-process decoding gives us no way
 * to abandon a decode that runs long; a subprocess does. Read per-call so it
 * can be lowered in tests. Set per-host via env.
 */
const timeoutMs = () => {
	const n = Number(process.env.NSFW_TRANSCODE_TIMEOUT_MS)
	return Number.isFinite(n) && n > 0 ? Math.floor(n) : 20_000
}

/**
 * Max ffmpeg processes in flight. classifier-host admits up to 50 images at
 * once, and spawning that many subprocesses simultaneously is wasteful. Well
 * above normal load, so throughput is unaffected.
 */
const NSFW_TRANSCODE_CONCURRENCY = (() => {
	const n = Number(process.env.NSFW_TRANSCODE_CONCURRENCY)
	return Number.isFinite(n) && n > 0 ? Math.floor(n) : 16
})()

logger(prefix, `transcode config NSFW_TRANSCODE_CONCURRENCY=${NSFW_TRANSCODE_CONCURRENCY} TIMEOUT_MS=${timeoutMs()}`)

/** ffmpeg exceeded the time limit and was killed. */
export class TranscodeTimeoutError extends Error {
	constructor(public readonly ms: number) {
		super(`ffmpeg transcode exceeded ${ms}ms and was killed`)
		this.name = 'TranscodeTimeoutError'
	}
}

/** ffmpeg ran and rejected the input. The buffer is not decodable. */
export class TranscodeError extends Error {
	constructor(message: string, public readonly stderr: string) {
		super(message)
		this.name = 'TranscodeError'
	}
}

const parseFfmpegErrorMessage = (stderr: string) => {
	const trimmed = stderr.trim()
	if (!trimmed) return 'ffmpeg failed'
	const lines = trimmed.split('\n').map(line => line.trim()).filter(Boolean)
	return lines[lines.length - 1] ?? trimmed
}

/* Bounded concurrency. release() hands its slot straight to the next waiter
   rather than decrementing, so a synchronous acquire() cannot jump the queue. */
let active = 0
const waiting: (() => void)[] = []

const acquire = (): Promise<void> => {
	if (active < NSFW_TRANSCODE_CONCURRENCY) {
		active++
		return Promise.resolve()
	}
	return new Promise<void>(resolve => waiting.push(resolve))
}

const release = () => {
	const next = waiting.shift()
	if (next) next()
	else active--
}

/*
 * `-frames:v 1` takes the first frame only, for animated inputs.
 * Scale to 299 (= NsfwTools IMAGE_SIZE, keep in step): preprocess() squashes to that
 * anyway, so same geometry for a fraction of the pixels, and preprocess() then skips
 * its resize. Stored, unfiltered PNG - deflate earns nothing at this size.
 * `-pix_fmt rgb24` forces 8-bit 3-channel output, so alpha removal is deterministic
 * and tf.node.decodeImage always gets something it can read.
 */
const ffmpegArgs = (input: string, output: string) => [
	'-nostdin',
	'-hide_banner',
	'-loglevel', 'error',
	'-i', input,
	'-frames:v', '1',
	'-vf', 'scale=299:299',
	'-pix_fmt', 'rgb24',
	'-f', 'image2',
	'-c:v', 'png',
	'-compression_level', '0',
	'-pred', 'none',
	'-y', output,
]

const runFfmpeg = (args: string[]) => new Promise<void>((resolve, reject) => {
	const stderrChunks: string[] = []
	let timedOut = false

	const child = spawn('ffmpeg', args, { stdio: ['ignore', 'ignore', 'pipe'] })

	const timer = setTimeout(() => {
		timedOut = true
		child.kill('SIGKILL')
	}, timeoutMs())

	child.stderr.on('data', chunk => stderrChunks.push(chunk.toString()))

	/* spawn failures (ENOENT - no ffmpeg binary) reject with the raw error on
	   purpose: an operational fault must stay loud, not be mapped to
	   'unsupported' and silently reclassify every non-native image as
	   undecodable. */
	child.on('error', err => {
		clearTimeout(timer)
		reject(err)
	})

	child.on('close', code => {
		clearTimeout(timer)
		if (timedOut) return reject(new TranscodeTimeoutError(timeoutMs()))
		if (code === 0) return resolve()
		const stderr = stderrChunks.join('')
		reject(new TranscodeError(parseFfmpegErrorMessage(stderr), stderr))
	})
})

/**
 * Name the input temp file after the content type, e.g. 'image/webp' -> 'in.webp'.
 * ffmpeg content-probes regardless, but an extension helps it pick a demuxer for
 * the container-based formats. Anything unrecognisable gets no extension rather
 * than a sanitised guess - `contentType` is untrusted, so it never reaches a path.
 */
const inputName = (contentType: string) => {
	const subtype = contentType.split('/')[1]?.toLowerCase() ?? ''
	return /^[a-z0-9]+$/.test(subtype) ? `in.${subtype}` : 'in'
}

/**
 * Decode an image buffer to PNG bytes via an ffmpeg subprocess. This is the only
 * decode path for every format tfjs-node cannot read natively.
 *
 * Isolation is the point. In-process decoding runs on the libuv threadpool,
 * which is small (4 slots by default) and shared - a decode that runs long
 * holds a slot the rest of the process needs, and there is no way to abandon
 * it. A subprocess is independently killable and bounded by a timeout, so the
 * cost of one bad input stays with that one input.
 *
 * Data moves via temp files in both directions, never stdin/stdout: writing
 * input while ffmpeg blocks writing output is a classic deadlock, and the
 * box-based formats (avif, heic) need to seek, which a pipe cannot do.
 */
export const transcodeToPng = async (pic: Buffer, contentType: string): Promise<Buffer> => {
	const queued = Date.now()
	await acquire()
	const acquired = Date.now()
	let ffmpegMs = 0
	try {
		const dir = await mkdtemp(path.join(os.tmpdir(), 'nsfw-decode-'))
		try {
			const input = path.join(dir, inputName(contentType))
			const output = path.join(dir, 'out.png')
			await writeFile(input, pic)
			const ran = Date.now()
			await runFfmpeg(ffmpegArgs(input, output))
			ffmpegMs = Date.now() - ran
			return await readFile(output)
		} finally {
			await rm(dir, { recursive: true, force: true })
		}
	} finally {
		logger(prefix, `transcode wait=${acquired - queued}ms ffmpeg=${ffmpegMs}ms slot=${Date.now() - acquired}ms waiting=${waiting.length}`)
		release()
	}
}
