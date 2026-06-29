/**
 * Side-effect module: must be imported BEFORE @tensorflow/tfjs-node.
 *
 * @tensorflow/tfjs-node (all published versions, incl. 4.22.0 / 4.23.0-rc.0)
 * imports `isArray` and `isNullOrUndefined` from node's `util` in
 * nodejs_kernel_backend and kernels/TopK. Both were removed in Node 22, so on
 * modern runtimes these calls throw at model-load / classify time. Reinstate
 * them here. Delete once upstream stops importing the removed util helpers.
 */

import util from 'util'

//@ts-expect-error - util.isNullOrUndefined is not defined in the type definitions
if (!util.isNullOrUndefined) {
	//@ts-expect-error - util.isNullOrUndefined is not defined in the type definitions
	util.isNullOrUndefined = (val: unknown): val is null | undefined =>
		val === null || val === undefined
}

if (!util.isArray) {
	util.isArray = Array.isArray
}
