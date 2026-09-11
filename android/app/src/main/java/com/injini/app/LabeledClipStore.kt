package com.injini.app

import android.content.Context
import android.os.Build
import java.io.File
import java.io.RandomAccessFile

/**
 * Saves a field-collected training clip as a playable WAV file plus one row
 * in a running manifest CSV, in the same column shape
 * prepare_engine_sounds.py already expects (path, label, class_name,
 * source_key, split) with the extra provenance a field recording needs on
 * top. The point of this file is that `adb pull` of one directory, plus a
 * script that assigns train/val by source_key, is the entire path from a
 * phone in Harare to a clip in the next Kaggle retrain — no separate export
 * step to design later.
 *
 * machine_id is the source_key: every clip from one physical machine is one
 * source, which is exactly the unit prepare_engine_sounds.py already splits
 * on to keep near-identical recordings out of both sides of a split.
 */
class LabeledClipStore(private val context: Context) {

    private val root = File(context.getExternalFilesDir(null), "InjiniLabeled")
    private val manifest = File(root, "manifest.csv")

    data class SaveResult(val wavFile: File, val corpusLabel: String)

    fun save(
        pcm: FloatArray,
        sampleRate: Int,
        machineId: String,
        corpusLabel: String,
        displayLabel: String,
        mechanicVerdict: String,
        captureSource: AudioCapture.Source,
    ): SaveResult {
        val safeMachine = machineId.trim().ifEmpty { "unknown" }.replace(Regex("[^A-Za-z0-9_-]"), "_")
        val safeLabel = corpusLabel.replace(Regex("[^A-Za-z0-9_ -]"), "_")
        val dir = File(root, safeLabel)
        dir.mkdirs()
        ensureManifestHeader()

        val ts = System.currentTimeMillis()
        val wav = File(dir, "${safeMachine}_$ts.wav")
        writeWav(wav, pcm, sampleRate)

        val row = listOf(
            wav.absolutePath,
            corpusLabel,
            corpusLabel,          // class_name mirrors label, matching prepare_engine_sounds.py's schema
            safeMachine,          // source_key: the machine, not the file — the whole point of the split
            "",                   // split: assigned later, on export, not on-device
            displayLabel,
            mechanicVerdict.replace(",", ";"),
            captureSource.name,
            Build.MANUFACTURER + " " + Build.MODEL,
            sampleRate.toString(),
            "%.2f".format(pcm.size.toDouble() / sampleRate),
            MachineRegistry.nowIso(),
        ).joinToString(",") { csvEscape(it) }
        manifest.appendText(row + "\n")

        return SaveResult(wav, corpusLabel)
    }

    fun clipCount(): Int {
        if (!manifest.exists()) return 0
        return (manifest.readLines().size - 1).coerceAtLeast(0)
    }

    private fun ensureManifestHeader() {
        if (manifest.exists()) return
        root.mkdirs()
        manifest.writeText(
            listOf(
                "path", "label", "class_name", "source_key", "split",
                "display_label", "mechanic_verdict", "capture_source",
                "device", "sample_rate", "duration_s", "recorded_at_utc",
            ).joinToString(",") + "\n"
        )
    }

    private fun csvEscape(s: String): String =
        if (s.contains(',') || s.contains('"') || s.contains('\n'))
            "\"" + s.replace("\"", "\"\"") + "\""
        else s

    /** Minimal mono 16-bit PCM WAV, no external dependency. */
    private fun writeWav(file: File, pcm: FloatArray, sampleRate: Int) {
        val byteRate = sampleRate * 2
        val dataSize = pcm.size * 2
        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0)
            fun le32(v: Int) { raf.write(byteArrayOf((v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte(), ((v shr 16) and 0xFF).toByte(), ((v shr 24) and 0xFF).toByte())) }
            fun le16(v: Int) { raf.write(byteArrayOf((v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte())) }

            raf.writeBytes("RIFF"); le32(36 + dataSize); raf.writeBytes("WAVE")
            raf.writeBytes("fmt "); le32(16); le16(1) /* PCM */; le16(1) /* mono */
            le32(sampleRate); le32(byteRate); le16(2) /* block align */; le16(16) /* bits/sample */
            raf.writeBytes("data"); le32(dataSize)
            val buf = ByteArray(pcm.size * 2)
            var p = 0
            for (s in pcm) {
                val v = (s.coerceIn(-1f, 1f) * 32767f).toInt()
                buf[p++] = (v and 0xFF).toByte()
                buf[p++] = ((v shr 8) and 0xFF).toByte()
            }
            raf.write(buf)
        }
    }
}
