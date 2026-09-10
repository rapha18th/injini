package com.injini.app

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin

/**
 * Diffs the Kotlin log-mel (the exact math from AudioFeatures, reimplemented
 * here without an Android Context so it runs on a plain JVM) against the Python
 * reference produced by src/features.py (EfficientAT's AugmentMelSTFT).
 *
 * Resources, written by the generator in the repo:
 *   test_audio_16k.pcm       int16 LE mono @ 16 kHz
 *   test_logmel_ref.bin      int32 rows, int32 cols, then rows*cols float32
 *
 * A silent mismatch here would quietly degrade every anomaly score in the
 * field with no crash, which is the failure this test exists to catch.
 */
class AudioFeaturesParityTest {

    private val res = File("src/test/resources")

    private fun readPcm16(f: File): FloatArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(f.length().toInt() / 2)
        for (i in out.indices) out[i] = b.short / 32768f
        return out
    }

    private fun readMatrix(f: File): Pair<IntArray, FloatArray> {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val rows = b.int; val cols = b.int
        val v = FloatArray(rows * cols) { b.float }
        return intArrayOf(rows, cols) to v
    }

    @Test
    fun kotlinLogMelMatchesPythonReference() {
        val pcm16k = readPcm16(File(res, "test_audio_16k.pcm"))
        val (shape, ref) = readMatrix(File(res, "test_logmel_ref.bin"))
        val (rows, cols) = shape

        val y32 = Ref.resampleTo32k(pcm16k, 16_000)
        val mel = Ref.logMel(y32)                       // (128, T)

        var maxAbs = 0.0
        var count = 0
        for (i in 0 until rows) {
            for (j in 0 until minOf(cols, mel[i].size)) {
                maxAbs = maxOf(maxAbs, abs(mel[i][j] - ref[i * cols + j]).toDouble())
                count++
            }
        }
        println("parity: compared $count cells, max abs diff $maxAbs")
        assertTrue("max abs diff $maxAbs exceeds 1e-3", maxAbs < 1e-3)
    }

    /**
     * Context-free copy of AudioFeatures' math. The only difference is the mel
     * filterbank is read straight from res/raw here rather than via Resources.
     */
    private object Ref {
        const val SR = 32_000
        const val N_FFT = 1024
        const val HOP = 320
        const val WIN = 800
        const val N_MELS = 128
        const val N_FREQS = N_FFT / 2 + 1
        const val PREEMPH = 0.97
        private val WINDOW = DoubleArray(N_FFT).also { w ->
            val pad = (N_FFT - WIN) / 2
            for (n in 0 until WIN) w[pad + n] = 0.5 - 0.5 * cos(2.0 * PI * n / (WIN - 1))
        }
        private val MEL: Array<DoubleArray> = run {
            val bytes = File("src/main/res/raw/mel_kaldi_128x513.bin").readBytes()
            val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            Array(N_MELS) { DoubleArray(N_FREQS) { b.float.toDouble() } }
        }

        fun resampleTo32k(y: FloatArray, origSr: Int): FloatArray {
            if (origSr == SR) return y
            val dur = y.size.toDouble() / origSr
            val nt = Math.round(dur * SR).toInt()
            val out = FloatArray(nt)
            val step = y.size.toDouble() / nt
            for (i in 0 until nt) {
                val x = i * step
                val i0 = x.toInt().coerceIn(0, y.size - 1)
                val i1 = (i0 + 1).coerceAtMost(y.size - 1)
                val f = (x - i0).toFloat()
                out[i] = y[i0] * (1 - f) + y[i1] * f
            }
            return out
        }

        private fun fft(re: DoubleArray, im: DoubleArray) {
            val n = re.size
            var j = 0
            for (i in 1 until n) {
                var bit = n shr 1
                while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
                j = j or bit
                if (i < j) {
                    val tr = re[i]; re[i] = re[j]; re[j] = tr
                    val ti = im[i]; im[i] = im[j]; im[j] = ti
                }
            }
            var len = 2
            while (len <= n) {
                val ang = -2.0 * PI / len
                val wr = cos(ang); val wi = sin(ang)
                var i = 0
                while (i < n) {
                    var cr = 1.0; var ci = 0.0
                    for (k in 0 until len / 2) {
                        val ur = re[i + k]; val ui = im[i + k]
                        val vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci
                        val vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr
                        re[i + k] = ur + vr; im[i + k] = ui + vi
                        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi
                        val nr = cr * wr - ci * wi; val ni = cr * wi + ci * wr
                        cr = nr; ci = ni
                    }
                    i += len
                }
                len = len shl 1
            }
        }

        fun logMel(y32k: FloatArray): Array<FloatArray> {
            val emph = DoubleArray(y32k.size - 1) { t -> (y32k[t + 1] - PREEMPH * y32k[t]).toDouble() }
            val pad = N_FFT / 2
            val p = DoubleArray(emph.size + 2 * pad)
            for (i in 0 until pad) p[i] = emph[pad - i]
            System.arraycopy(emph, 0, p, pad, emph.size)
            for (k in 0 until pad) p[pad + emph.size + k] = emph[emph.size - 2 - k]
            val nFrames = 1 + (p.size - N_FFT) / HOP
            val out = Array(N_MELS) { FloatArray(nFrames) }
            for (f in 0 until nFrames) {
                val re = DoubleArray(N_FFT); val im = DoubleArray(N_FFT)
                val s = f * HOP
                for (n in 0 until N_FFT) re[n] = p[s + n] * WINDOW[n]
                fft(re, im)
                for (i in 0 until N_MELS) {
                    var acc = 0.0
                    val fb = MEL[i]
                    for (k in 0 until N_FREQS) acc += fb[k] * (re[k] * re[k] + im[k] * im[k])
                    out[i][f] = ((ln(acc + 1e-5) + 4.5) / 5.0).toFloat()
                }
            }
            return out
        }
    }
}
