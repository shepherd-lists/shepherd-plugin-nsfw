import { expect } from 'chai'
import { describe } from 'mocha'
import NsfwFilter from '../src/index'
import { FilterResult } from 'shepherd-plugin-interfaces'
import fs from 'fs/promises'
import os from 'os'
import path from 'path'
import sharp from 'sharp'

/* Builds a temp dir of frame-<n>.png files. `sources` is a list of source image
   buffers; each becomes frame-1.png, frame-2.png, ... in order, resized to a
   non-299 size to exercise the resize path. */
const makeFramesDir = async (sources: Buffer[]): Promise<string> => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'nsfw-frames-'))
	await Promise.all(sources.map((buf, i) =>
		sharp(buf).resize(640, 360).png().toFile(path.join(dir, `frame-${i + 1}.png`))
	))
	return dir
}

describe('checkImageDir tests', () => {
	let clean: Buffer
	let flagged: Buffer

	before('loads the model', async function () {
		this.timeout(0)
		await NsfwFilter.init()
		clean = await fs.readFile('./test/assets/image.jpeg')
		flagged = await fs.readFile('./test/assets/oci0s0Y-u-vIewMCHr1XgMX07if2KkBfKnRZD2sraps.jpg')
	})

	it('classifies a clean video as not-flagged', async () => {
		const dir = await makeFramesDir(Array(10).fill(clean))
		try {
			const res = await NsfwFilter.checkImageDir!(dir, 'image/png', 'clean-vid-txid')
			expect(res.flagged).false
		} finally {
			await fs.rm(dir, { recursive: true, force: true })
		}
	}).timeout(0)

	it('flags a video when a frame is flagged (first flagged wins)', async () => {
		// clean frames with one flagged frame in the middle
		const sources = [clean, clean, clean, flagged, clean, clean]
		const dir = await makeFramesDir(sources)
		try {
			const res = await NsfwFilter.checkImageDir!(dir, 'image/png', 'flagged-vid-txid') as FilterResult
			expect(res.flagged).true
			expect(res.top_score_name).eq('Sexy')
		} finally {
			await fs.rm(dir, { recursive: true, force: true })
		}
	}).timeout(0)

	it('handles a batch spanning multiple predicts (count > batch size)', async () => {
		// 20 frames forces multiple flushes at the default batch size of 16
		const dir = await makeFramesDir(Array(20).fill(clean))
		try {
			const res = await NsfwFilter.checkImageDir!(dir, 'image/png', 'many-frames-txid')
			expect(res.flagged).false
		} finally {
			await fs.rm(dir, { recursive: true, force: true })
		}
	}).timeout(0)

	it('throws when the frames dir is empty', async () => {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'nsfw-empty-'))
		try {
			await NsfwFilter.checkImageDir!(dir, 'image/png', 'empty-txid')
			expect.fail('expected checkImageDir to throw on empty dir')
		} catch (e) {
			expect((e as Error).message).match(/no frame-\*\.png/)
		} finally {
			await fs.rm(dir, { recursive: true, force: true })
		}
	}).timeout(0)
})
