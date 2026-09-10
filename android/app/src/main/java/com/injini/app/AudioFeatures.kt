package com.injini.app

import android.content.Context
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin

/**
 * Kotlin port of src/features.py — EfficientAT's mel spec, exactly.
 *
 *   pre-emphasis 0.97, STFT n_fft=1024 / hop=320 / win_length=800 (symmetric
 *   Hann zero-padded to n_fft, centered/reflect), 128 Kaldi mel bands
 *   0-15000 Hz, log(mel + 1e-5), then (x + 4.5) / 5.
 *
 * The Kaldi filterbank is not re-derived here: it is shipped as
 * res/raw/mel_kaldi_128x513.bin (128x513 little-endian float32, the same
 * matrix src/features.py caches) and loaded at construction. Parity with the
 * Python reference is checked in AudioFeaturesParityTest.
 *
 * The radix-2 FFT is carried over verbatim from SiloSense, where it is already
 * parity-tested to 4.2e-7.
 */
class AudioFeatures(context: Context) {

    companion object {
        const val SR = 32_000
        const val N_FFT = 1024
        const val HOP = 320
        const val WIN_LENGTH = 800
        const val N_MELS = 128
        const val N_FREQS = N_FFT / 2 + 1            // 513
        const val PREEMPH = 0.97
        const val LOG_OFFSET = 1e-5
        const val NORM_ADD = 4.5
        const val NORM_DIV = 5.0

        const val CLIP_SECONDS = 10.0
        const val CLIP_SAMPLES = (SR * CLIP_SECONDS).toInt()          // 320000
        const val N_FRAMES = 1 + CLIP_SAMPLES / HOP                    // 1001

        // Symmetric Hann (periodic=false), zero-padded to N_FFT and centered.
        private val WINDOW: DoubleArray = DoubleArray(N_FFT).also { w ->
            val padLeft = (N_FFT - WIN_LENGTH) / 2
            for (n in 0 until WIN_LENGTH) {
                w[padLeft + n] = 0.5 - 0.5 * cos(2.0 * PI * n / (WIN_LENGTH - 1))
            }
        }
    }

    /** (128, 513) Kaldi mel filterbank, loaded from res/raw. */
    private val mel: Array<DoubleArray> = run {
        val bytes = context.resources.openRawResource(
            context.resources.getIdentifier("mel_kaldi_128x513", "raw", context.packageName)
        ).use { it.readBytes() }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        Array(N_MELS) { DoubleArray(N_FREQS) { buf.float.toDouble() } }
    }

    /** Resample to 32 kHz (naive linear interp, matches features.resample_linear). */
    fun resampleTo32k(y: FloatArray, origSr: Int): FloatArray {
        if (origSr == SR || y.isEmpty()) return y
        val duration = y.size.toDouble() / origSr
        val nTarget = maxOf(1, Math.round(duration * SR).toInt())
        val out = FloatArray(nTarget)
        val step = y.size.toDouble() / nTarget
        for (i in 0 until nTarget) {
            val x = i * step
            val i0 = x.toInt().coerceIn(0, y.size - 1)
            val i1 = (i0 + 1).coerceAtMost(y.size - 1)
            val frac = (x - i0).toFloat()
            out[i] = y[i0] * (1 - frac) + y[i1] * frac
        }
        return out
    }

    private fun fixedLength(y: FloatArray): FloatArray {
        if (y.size == CLIP_SAMPLES) return y
        if (y.size > CLIP_SAMPLES) return y.copyOf(CLIP_SAMPLES)
        return FloatArray(CLIP_SAMPLES).also { System.arraycopy(y, 0, it, 0, y.size) }
    }

    /** out[t] = x[t+1] - 0.97*x[t], length N-1 (matches torch conv1d kernel [-.97, 1]). */
    private fun preemphasis(y: FloatArray): DoubleArray =
        DoubleArray(y.size - 1) { t -> (y[t + 1] - PREEMPH * y[t]) }

    private fun reflectPad(y: DoubleArray, pad: Int): DoubleArray {
        val n = y.size
        val out = DoubleArray(n + 2 * pad)
        for (i in 0 until pad) out[i] = y[pad - i]
        System.arraycopy(y, 0, out, pad, n)
        for (j in 0 until pad) out[pad + n + j] = y[n - 2 - j]
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
                var curWr = 1.0; var curWi = 0.0
                for (k in 0 until len / 2) {
                    val ur = re[i + k]; val ui = im[i + k]
                    val vr = re[i + k + len / 2] * curWr - im[i + k + len / 2] * curWi
                    val vi = re[i + k + len / 2] * curWi + im[i + k + len / 2] * curWr
                    re[i + k] = ur + vr; im[i + k] = ui + vi
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi
                    val nwr = curWr * wr - curWi * wi
                    val nwi = curWr * wi + curWi * wr
                    curWr = nwr; curWi = nwi
                }
                i += len
            }
            len = len shl 1
        }
    }

    private fun framePower(frame: DoubleArray): DoubleArray {
        val re = DoubleArray(N_FFT)
        val im = DoubleArray(N_FFT)
        for (n in 0 until N_FFT) re[n] = frame[n] * WINDOW[n]
        fft(re, im)
        return DoubleArray(N_FREQS) { k -> re[k] * re[k] + im[k] * im[k] }
    }

    /**
     * PCM (mono float, roughly [-1,1], already at 32 kHz) -> (N_MELS, N_FRAMES)
     * log-mel normalised the EfficientAT way. Matches src/features.py logmel(fixed=True).
     */
    fun logMel(y32k: FloatArray): Array<FloatArray> {
        val emph = preemphasis(fixedLength(y32k))
        val padded = reflectPad(emph, N_FFT / 2)
        val nFrames = 1 + (padded.size - N_FFT) / HOP

        val out = Array(N_MELS) { FloatArray(N_FRAMES) }
        val lastCol = FloatArray(N_MELS)
        for (f in 0 until nFrames) {
            val start = f * HOP
            val frame = DoubleArray(N_FFT) { k -> padded[start + k] }
            val power = framePower(frame)
            for (i in 0 until N_MELS) {
                var s = 0.0
                val fb = mel[i]
                for (k in 0 until N_FREQS) s += fb[k] * power[k]
                val v = ((ln(s + LOG_OFFSET) + NORM_ADD) / NORM_DIV).toFloat()
                if (f < N_FRAMES) out[i][f] = v
                lastCol[i] = v
            }
        }
        // Pad short clips by repeating the last frame (features.py: mode="edge").
        for (f in nFrames until N_FRAMES) for (i in 0 until N_MELS) out[i][f] = lastCol[i]
        return out
    }

    /** Convenience: raw PCM at [origSr] -> log-mel. */
    fun logMelFromPcm(y: FloatArray, origSr: Int): Array<FloatArray> =
        logMel(resampleTo32k(y, origSr))
}
