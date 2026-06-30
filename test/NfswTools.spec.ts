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

	describe('sharp-decoded formats', () => {

		/* sharp re-encodes the known-safe jpeg asset into each format, so we can
		   assert these formerly-"unsupported" types now actually get classified. */
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

	})

})