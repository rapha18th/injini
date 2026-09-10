package com.injini.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONArray
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Wraps an ONNX Runtime session around the frozen EfficientAT embedder
 * (assets/injini_mn10_as_int8.onnx by default). Input: (1, 1, 128, N_FRAMES)
 * log-mel. Output: an L2-normalised embedding vector.
 *
 * The model here is only the feature extractor. The anomaly decision is in
 * AnomalyScorer and never touches ONNX. Benchmark and execution-provider
 * profiling follow SiloSense exactly: time only session.run(), warm up and
 * discard, read the profile trace for which provider ran each node.
 */
class InjiniEmbedder(
    context: Context,
    modelAsset: String = "injini_mn10_as_int8.onnx",
) : AutoCloseable {

    data class BenchmarkResult(val minMs: Double, val avgMs: Double, val maxMs: Double, val runs: Int)

    val configuredProviders: String
    val loadTimeMs: Double
    val embedDim: Int

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val profilePrefix: String
    private var profilingEnded = false

    init {
        val t0 = System.nanoTime()
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val opts = OrtSession.SessionOptions()
        val providers = mutableListOf<String>()
        try { opts.addNnapi(); providers += "NNAPI" } catch (_: OrtException) {}
        try { opts.addXnnpack(emptyMap()); providers += "XNNPACK" } catch (_: OrtException) {}
        providers += "CPU"
        configuredProviders = providers.joinToString("+")

        profilePrefix = File(context.cacheDir, "ort_${modelAsset.substringBeforeLast('.')}_").absolutePath
        try { opts.enableProfiling(profilePrefix) } catch (_: OrtException) {}

        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()
        loadTimeMs = (System.nanoTime() - t0) / 1_000_000.0

        val outInfo = session.outputInfo.values.first().info as ai.onnxruntime.TensorInfo
        embedDim = outInfo.shape.last().toInt().let { if (it > 0) it else 960 }
    }

    /** (128, N_FRAMES) log-mel -> L2-normalised embedding. */
    fun embed(logmel: Array<FloatArray>): FloatArray {
        val flat = FloatArray(AudioFeatures.N_MELS * AudioFeatures.N_FRAMES)
        for (i in logmel.indices) {
            System.arraycopy(logmel[i], 0, flat, i * AudioFeatures.N_FRAMES, AudioFeatures.N_FRAMES)
        }
        val shape = longArrayOf(1, 1, AudioFeatures.N_MELS.toLong(), AudioFeatures.N_FRAMES.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val vec = (result[0].value as Array<FloatArray>)[0]
                return l2(vec)
            }
        }
    }

    fun timedEmbed(logmel: Array<FloatArray>): Pair<FloatArray, Double> {
        val start = System.nanoTime()
        val e = embed(logmel)
        return e to (System.nanoTime() - start) / 1_000_000.0
    }

    fun warmUp(count: Int = 5) {
        val dummy = Array(AudioFeatures.N_MELS) { FloatArray(AudioFeatures.N_FRAMES) }
        repeat(count) { embed(dummy) }
    }

    fun benchmark(logmel: Array<FloatArray>, warmup: Int = 5, timed: Int = 20): BenchmarkResult {
        repeat(warmup) { embed(logmel) }
        val ms = DoubleArray(timed)
        for (i in 0 until timed) {
            val s = System.nanoTime()
            embed(logmel)
            ms[i] = (System.nanoTime() - s) / 1_000_000.0
        }
        return BenchmarkResult(ms.min(), ms.average(), ms.max(), timed)
    }

    fun stressRun(logmel: Array<FloatArray>, durationMs: Long): Int {
        val end = System.nanoTime() + durationMs * 1_000_000
        var n = 0
        while (System.nanoTime() < end) { embed(logmel); n++ }
        return n
    }

    /** One-shot: run [count] real inferences with profiling on, then report per-node provider counts. */
    fun diagnoseExecutionProviders(logmel: Array<FloatArray>, count: Int = 5): String {
        if (profilingEnded) return "profiling already ended"
        repeat(count) { embed(logmel) }
        val path = try { session.endProfiling() } catch (e: OrtException) {
            return "profiling unavailable (${e.message})"
        } finally { profilingEnded = true }
        return try {
            val events = JSONArray(File(path).readText())
            val counts = linkedMapOf<String, Int>()
            for (i in 0 until events.length()) {
                val a = events.getJSONObject(i).optJSONObject("args") ?: continue
                val p = a.optString("provider", "")
                if (p.isNotEmpty()) counts[p] = (counts[p] ?: 0) + 1
            }
            if (counts.isEmpty()) "no per-node provider field in trace"
            else counts.entries.joinToString(", ") { "${it.key}=${it.value}" }
        } catch (e: Exception) { "profile written but unparsed (${e.message})" }
    }

    override fun close() = session.close()

    private fun l2(v: FloatArray): FloatArray {
        var s = 0.0
        for (x in v) s += x * x
        val n = (sqrt(s) + 1e-12).toFloat()
        return FloatArray(v.size) { v[it] / n }
    }
}
