/**
 * nsfwjs rating system:
 * 
 * drawings - safe for work drawings (including anime)
 * neutral - safe for work neutral images
 * sexy - sexually explicit images, not pornography
 * hentai - hentai and pornographic drawings
 * porn - pornographic images, sexual acts
 * 
 * Supported formats: BMP/JPEG/PNG are decoded natively by tfjs-node. Every other
 * image type is transcoded to PNG by an ffmpeg subprocess first (see
 * transcode.ts), so coverage follows whatever the ffmpeg build supports - note
 * ffmpeg has no SVG rasteriser, so SVG is not classified. Input ffmpeg cannot
 * decode returns data_reason: 'unsupported'.
 */

import { FilterPluginInterface } from "shepherd-plugin-interfaces";
import { NsfwTools } from "./NsfwTools";

const NsfwjsPlugin: FilterPluginInterface = {
	init: NsfwTools.init,
	checkImage: NsfwTools.checkImage,
	checkImageDir: NsfwTools.checkImageDir,
}

export default NsfwjsPlugin;