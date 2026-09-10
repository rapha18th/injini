package com.injini.app

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.sqrt

/**
 * On-device anomaly score: how far a clip's embedding sits from the healthy
 * recordings of this machine.
 *
 * The phone uses two cheap scores and combines them:
 *   - kNN cosine distance to the k nearest enrolled healthy embeddings
 *   - diagonal Mahalanobis (per-dimension mean and variance)
 *
 * The full-covariance Mahalanobis used in the DCASE evaluation needs a DxD
 * inverse; on the phone the diagonal form is used instead, which needs no
 * matrix inverse and is stable from a handful of enrolment clips. Enrolment =
 * record N healthy clips of one machine, embed them, call [enroll].
 *
 * Bounds for the three-tier verdict are calibrated from validation data, not a
 * round number; [Bounds] carries defaults that the training run overwrites.
 */
class AnomalyScorer private constructor(
    private val refs: Array<FloatArray>,     // enrolled healthy embeddings (L2-normalised)
    private val mean: FloatArray,
    private val invVar: FloatArray,          // 1 / (variance + eps) per dim
    private val k: Int,
    val bounds: Bounds,
) {

    data class Bounds(val healthy: Float, val watch: Float) {
        companion object { val DEFAULT = Bounds(0.55f, 0.75f) }
    }

    enum class Tier { HEALTHY, WATCH, CHECK }

    data class Result(
        val score: Float,          // 0 = identical to healthy, higher = more unlike
        val knnDistance: Float,
        val mahalanobis: Float,
        val tier: Tier,
    )

    fun score(embedding: FloatArray): Result {
        val e = l2(embedding)

        // kNN cosine distance
        val sims = FloatArray(refs.size) { dot(e, refs[it]) }
        sims.sort()
        val kk = minOf(k, sims.size)
        var s = 0f
        for (i in 0 until kk) s += sims[sims.size - 1 - i]
        val knn = 1f - s / kk

        // diagonal Mahalanobis, scaled to a comparable range
        var m = 0.0
        for (i in e.indices) {
            val d = e[i] - mean[i]
            m += d * d * invVar[i]
        }
        val maha = (sqrt(m) / sqrt(e.size.toDouble())).toFloat()

        val combined = 0.5f * knn + 0.5f * maha
        val tier = when {
            combined < bounds.healthy -> Tier.HEALTHY
            combined < bounds.watch -> Tier.WATCH
            else -> Tier.CHECK
        }
        return Result(combined, knn, maha, tier)
    }

    fun toJson(): String {
        val o = JSONObject()
        o.put("k", k)
        o.put("healthy_bound", bounds.healthy.toDouble())
        o.put("watch_bound", bounds.watch.toDouble())
        o.put("mean", JSONArray(mean.map { it.toDouble() }))
        o.put("inv_var", JSONArray(invVar.map { it.toDouble() }))
        val r = JSONArray()
        for (row in refs) r.put(JSONArray(row.map { it.toDouble() }))
        o.put("refs", r)
        return o.toString()
    }

    companion object {
        private const val EPS = 1e-6f

        fun enroll(healthy: List<FloatArray>, k: Int = 4, bounds: Bounds = Bounds.DEFAULT): AnomalyScorer {
            require(healthy.size >= 2) { "need at least 2 healthy clips to enrol" }
            val d = healthy[0].size
            val refs = Array(healthy.size) { l2(healthy[it]) }

            val mean = FloatArray(d)
            for (r in refs) for (i in 0 until d) mean[i] += r[i]
            for (i in 0 until d) mean[i] /= refs.size

            val varr = FloatArray(d)
            for (r in refs) for (i in 0 until d) {
                val x = r[i] - mean[i]; varr[i] += x * x
            }
            val invVar = FloatArray(d) { 1f / (varr[it] / refs.size + EPS) }

            return AnomalyScorer(refs, mean, invVar, k, bounds)
        }

        fun fromJson(s: String): AnomalyScorer {
            val o = JSONObject(s)
            val meanA = o.getJSONArray("mean"); val ivA = o.getJSONArray("inv_var")
            val mean = FloatArray(meanA.length()) { meanA.getDouble(it).toFloat() }
            val invVar = FloatArray(ivA.length()) { ivA.getDouble(it).toFloat() }
            val rA = o.getJSONArray("refs")
            val refs = Array(rA.length()) { i ->
                val row = rA.getJSONArray(i)
                FloatArray(row.length()) { row.getDouble(it).toFloat() }
            }
            return AnomalyScorer(
                refs, mean, invVar, o.optInt("k", 4),
                Bounds(o.optDouble("healthy_bound", 0.55).toFloat(),
                       o.optDouble("watch_bound", 0.75).toFloat()),
            )
        }

        private fun l2(v: FloatArray): FloatArray {
            var s = 0.0
            for (x in v) s += x * x
            val n = (sqrt(s) + 1e-12).toFloat()
            return FloatArray(v.size) { v[it] / n }
        }

        private fun dot(a: FloatArray, b: FloatArray): Float {
            var s = 0f
            for (i in a.indices) s += a[i] * b[i]
            return s
        }
    }
}
