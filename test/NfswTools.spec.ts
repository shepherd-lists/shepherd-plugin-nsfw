import { expect } from 'chai'
import { describe } from 'mocha'
import NsfwFilter from '../src/index'
import fs from 'fs/promises'
import sharp, { Sharp } from 'sharp'
import { FilterErrorResult, FilterResult } from 'shepherd-plugin-interfaces'

describe('NsfwTools tests', ()=>{
	before('loads the model', async function(){
		this.timeout(0)
		await NsfwFilter.init()
	})

	it('oversized png', async()=>{
		const pic = await fs.readFile('./test/assets/0Hycn44ITAICfn0YbQP1eg3IMueuf5LVKpUAbYiAJYs.png')
		const res = await NsfwFilter.checkImage(pic,'image/png', '0Hycn44ITAICfn0YbQP1eg3IMueuf5LVKpUAbYiAJYs')
		expect(res.flagged).undefined
		const resErr = res as FilterErrorResult
		expect(resErr.data_reason).eq('oversized')
	}).timeout(0)


	it('8HcUVJMAdb3HG9XWBLdwpFbCEtc-PmQmrJM-WvPfUcQ', async () => {
		const pic = await fs.readFile('./test/assets/8HcUVJMAdb3HG9XWBLdwpFbCEtc-PmQmrJM-WvPfUcQ.png')
		const res = await NsfwFilter.checkImage(pic, 'image/png', '8HcUVJMAdb3HG9XWBLdwpFbCEtc-PmQmrJM-WvPfUcQ')
		console.log(res)

		if(res.flagged !== undefined){
			console.log('On big dev machine you should have seen "Allocation of XXXXX exceeds 10% of free system memory" errors and system hanging')
			expect(res.flagged).false
		}else{
			console.log('test system, ran out of memory')
			expect((res as FilterErrorResult).data_reason).eq('oversized')
		}
	}).timeout(0)

	describe('ffmpeg-decoded formats', () => {

		/* sharp is a devDependency now - a fixture generator only. It re-encodes the
		   known-safe jpeg asset into each format so we can assert these non-native
		   types still get classified through the ffmpeg path. */
		const formats: { name: string; contentType: string; encode: (s: Sharp) => Sharp }[] = [
			{ name: 'webp', contentType: 'image/webp', encode: s => s.webp() },
			{ name: 'tiff', contentType: 'image/tiff', encode: s => s.tiff() },
			{ name: 'gif',  contentType: 'image/gif',  encode: s => s.gif() },
		]

		for (const { name, contentType, encode } of formats) {
			it(`classifies ${name} instead of returning 'unsupported'`, async () => {
				const jpeg = await fs.readFile('./test/assets/image.jpeg')
				const pic = await encode(sharp(jpeg)).toBuffer()

				const res = await NsfwFilter.checkImage(pic, contentType, `fake-${name}-txid`)

				expect((res as FilterErrorResult).data_reason).not.eq('unsupported')
				expect(res.flagged).to.be.a('boolean')
				expect((res as FilterResult).flagged).false // the safe asset should not be flagged
			}).timeout(0)
		}

		it("returns 'unsupported' for an undecodable buffer", async () => {
			const garbage = Buffer.from('not an image at all', 'utf8')
			const res = await NsfwFilter.checkImage(garbage, 'image/x-unknown', 'fake-garbage-txid')

			expect(res.flagged).undefined
			expect((res as FilterErrorResult).data_reason).eq('unsupported')
		}).timeout(0)

		/* Accepted coverage loss, pinned here so it is a deliberate decision rather
		   than a surprise: ffmpeg has an svg_pipe demuxer but no SVG rasteriser, in
		   any version, so svg cannot be classified and routes onward instead. */
		it("returns 'unsupported' for svg", async () => {
			const svg = Buffer.from(
				'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
				+ '<rect width="64" height="64" fill="#888"/></svg>',
				'utf8',
			)
			const res = await NsfwFilter.checkImage(svg, 'image/svg+xml', 'fake-svg-txid')

			expect(res.flagged).undefined
			expect((res as FilterErrorResult).data_reason).eq('unsupported')
		}).timeout(0)

	})

	describe('runtime dependencies', () => {

		/* The decoder libraries must not be reachable from the deployed image at all -
		   an in-process decode that cannot be abandoned is the failure this design
		   exists to avoid. sharp belongs in devDependencies (fixture generation);
		   a dependency's devDependencies are never installed, which is what keeps it
		   out of the runtime. Guard against it being moved back by a later edit. */
		it('does not ship sharp as a runtime dependency', () => {
			const pkg = require('../package.json')

			expect(pkg.dependencies).to.not.have.property('sharp')
			expect(pkg.devDependencies).to.have.property('sharp')
		})

	})

	/* Non-native formats are transcoded to png by an ffmpeg subprocess so the decode
	   is isolated and time-bounded. The format cases above are the happy path; these
	   cover the two failure branches. */
	describe('transcode failure handling', () => {

		it("returns 'unsupported' when ffmpeg rejects the buffer", async () => {
			// valid RIFF/WEBP header, truncated body -> ffmpeg exits non-zero
			const truncated = Buffer.concat([
				Buffer.from('RIFF'),
				Buffer.from([0x24, 0x00, 0x00, 0x00]),
				Buffer.from('WEBPVP8 '),
				Buffer.from([0x18, 0x00, 0x00, 0x00]),
			])

			const res = await NsfwFilter.checkImage(truncated, 'image/webp', 'fake-truncated-webp')

			expect(res.flagged).undefined
			expect((res as FilterErrorResult).data_reason).eq('unsupported')
		}).timeout(0)

		it("returns 'unsupported' when the transcode times out", async () => {
			const jpeg = await fs.readFile('./test/assets/image.jpeg')
			const pic = await sharp(jpeg).webp().toBuffer()

			// 1ms is shorter than ffmpeg's own startup, so this always trips the
			// SIGKILL path.
			const prev = process.env.NSFW_TRANSCODE_TIMEOUT_MS
			process.env.NSFW_TRANSCODE_TIMEOUT_MS = '1'
			try {
				const res = await NsfwFilter.checkImage(pic, 'image/webp', 'fake-timeout-webp')

				expect(res.flagged).undefined
				expect((res as FilterErrorResult).data_reason).eq('unsupported')
			} finally {
				if (prev === undefined) delete process.env.NSFW_TRANSCODE_TIMEOUT_MS
				else process.env.NSFW_TRANSCODE_TIMEOUT_MS = prev
			}
		}).timeout(0)

		it('still classifies webp after a timeout (no leaked concurrency slot)', async () => {
			const jpeg = await fs.readFile('./test/assets/image.jpeg')
			const pic = await sharp(jpeg).webp().toBuffer()

			const res = await NsfwFilter.checkImage(pic, 'image/webp', 'fake-after-timeout-webp')

			expect(res.flagged).to.be.a('boolean')
			expect((res as FilterResult).flagged).false
		}).timeout(0)

	})

})