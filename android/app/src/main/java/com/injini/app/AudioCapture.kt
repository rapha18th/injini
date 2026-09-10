package com.injini.app

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Records mono 16 kHz PCM from the least-processed source Android will give us.
 *
 * Default capture applies AGC, noise suppression and echo cancellation, all
 * tuned for speech and all destructive to the stationary periodic content that
 * carries a machine's condition. We ask for UNPROCESSED first, fall back to
 * VOICE_RECOGNITION (no AGC on most devices), and record which one we got so
 * the result can say so.
 *
 * 16 kHz here; AudioFeatures resamples to 32 kHz for the embedder. This keeps
 * the pipeline aligned with SiloSense and keeps knock detection (5-7 kHz)
 * honestly out of scope.
 */
class AudioCapture {

    enum class Source { UNPROCESSED, VOICE_RECOGNITION, MIC }

    data class Recording(val pcm: FloatArray, val sampleRate: Int, val source: Source)

    companion object {
        const val SAMPLE_RATE = 16_000
    }

    fun record(seconds: Double): Recording {
        val (record, source) = open()
        val total = (SAMPLE_RATE * seconds).toInt()
        val out = FloatArray(total)
        val buf = ByteArray(4096)
        record.startRecording()
        var written = 0
        try {
            while (written < total) {
                val n = record.read(buf, 0, buf.size)
                if (n <= 0) break
                val sb = ByteBuffer.wrap(buf, 0, n).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                var i = 0
                while (i < n / 2 && written < total) {
                    out[written++] = sb.get(i) / 32768f
                    i++
                }
            }
        } finally {
            record.stop()
            record.release()
        }
        return Recording(out, SAMPLE_RATE, source)
    }

    private fun open(): Pair<AudioRecord, Source> {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(4096)
        val order = listOf(
            MediaRecorder.AudioSource.UNPROCESSED to Source.UNPROCESSED,
            MediaRecorder.AudioSource.VOICE_RECOGNITION to Source.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.MIC to Source.MIC,
        )
        for ((src, tag) in order) {
            @Suppress("MissingPermission")
            val ar = AudioRecord(
                src, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, minBuf * 2
            )
            if (ar.state == AudioRecord.STATE_INITIALIZED) return ar to tag
            ar.release()
        }
        error("could not open any audio source")
    }
}
