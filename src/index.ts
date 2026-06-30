/**
 * nsfwjs rating system:
 * 
 * drawings - safe for work drawings (including anime)
 * neutral - safe for work neutral images
 * sexy - sexually explicit images, not pornography
 * hentai - hentai and pornographic drawings
 * porn - pornographic images, sexual acts
 * 
 * Supported formats: BMP/JPEG/PNG are decoded natively by tfjs-node; any other
 * image type (WebP, AVIF, TIFF, SVG, HEIC, ...) is decoded via sharp.
 * Undecodable input returns data_reason: 'unsupported'.
 */

import { FilterPluginInterface } from "shepherd-plugin-interfaces";
import { NsfwTools } from "./NsfwTools";

const NsfwjsPlugin: FilterPluginInterface = {
	init: NsfwTools.init,
	checkImage: NsfwTools.checkImage,
}

export default NsfwjsPlugin;