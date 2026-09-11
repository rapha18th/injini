package com.injini.app

import android.content.Context
import android.os.Build
import java.io.File
import java.io.RandomAccessFile

/**
 * Where every recording this app makes ends up as real training material, in
 * the same column shape prepare_engine_sounds.py already expects (path,
 * label, class_name, source_key, split) plus the provenance a field
 * recording needs on top. The point is that `adb pull` of one directory,
 * plus a script that assigns train/val by source_key, is the entire path
 * from a phone in the field to a clip in the next retrain.
 *
 * Two ways a clip gets in:
 *
 *   save()         An Enrol clip. It is healthy by construction — the whole
 *                  point of enrolling is recording the machine while it runs
 *                  well — so it is written straight to the corpus under the
 *                  "Normal" label, no extra button and no extra decision.
 *   savePending()  A Check clip. What Check actually knows at record time is
 *                  the on-device anomaly score, not a verdict from a person
 *                  who has opened the machine up. The clip is saved with
 *                  that score as a hint and queued in pending.csv; a label
 *                  is attached later, whenever the real answer is known, via
 *                  [confirmPending]. This is the normal order events happen
 *                  in: you hear something, then someone finds out what it
 *                  was.
 *
 * machine_id is the source_key: every clip from one physical machine is one
 * source, which is exactly the unit prepare_engine_sounds.py already splits
 * on. engine_type and category ride along on every row because a rod knock
 * does not sound the same on a diesel and a petrol engine, and a future
 * model needs to know which population a clip came from.
 */
class LabeledClipStore(private val context: Context) {

    data class PendingClip(
        val pendingId: String,
        val path: String,
        val machineId: String,
        val engineType: String,
        val category: String,
        val provisionalTier: String,
        val captureSource: String,
        val recordedAtUtc: String,
    )

    private val root = File(context.getExternalFilesDir(null), "InjiniLabeled")
    private val manifest = File(root, "manifest.csv")
    private val pendingDir = File(root, "_pending")
    private val pendingManifest = File(pendingDir, "pending.csv")

    private val manifestHeader = listOf(
        "path", "label", "class_name", "source_key", "split",
        "display_label", "mechanic_verdict", "capture_source",
        "device", "sample_rate", "duration_s", "recorded_at_utc",
        "engine_type", "category", "benchmark_outcome",
    )
    private val pendingHeader = listOf(
        "pending_id", "path", "machine_id", "engine_type", "category",
        "provisional_tier", "capture_source", "device", "sample_rate",
        "duration_s", "recorded_at_utc",
    )

    // ------------------------------------------------------------- confirmed

    fun save(
        pcm: FloatArray,
        sampleRate: Int,
        machineId: String,
        engineType: String,
        category: String,
        corpusLabel: String,
        displayLabel: String,
        mechanicVerdict: String,
        captureSource: AudioCapture.Source,
    ): File {
        val safeMachine = safe(machineId)
        val safeLabel = corpusLabel.replace(Regex("[^A-Za-z0-9_ -]"), "_")
        val dir = File(root, safeLabel)
        dir.mkdirs()
        ensureHeader(manifest, manifestHeader)

        val wav = File(dir, "${safeMachine}_${System.currentTimeMillis()}.wav")
        writeWav(wav, pcm, sampleRate)

        appendRow(manifest, listOf(
            wav.absolutePath, corpusLabel, corpusLabel, safeMachine, "",
            displayLabel, mechanicVerdict, captureSource.name,
            deviceName(), sampleRate.toString(), durationOf(pcm, sampleRate),
            MachineRegistry.nowIso(), engineType, category,
            "", // no on-device prediction to benchmark at enrolment time
        ))
        return wav
    }

    // -------------------------------------------------------------- pending

    /** Saves a Check recording with only a provisional (model) tier, to be labelled later. */
    fun savePending(
        pcm: FloatArray,
        sampleRate: Int,
        machineId: String,
        engineType: String,
        category: String,
        provisionalTier: String,
        captureSource: AudioCapture.Source,
    ): PendingClip {
        pendingDir.mkdirs()
        ensureHeader(pendingManifest, pendingHeader)

        val id = "${safe(machineId)}_${System.currentTimeMillis()}"
        val wav = File(pendingDir, "$id.wav")
        writeWav(wav, pcm, sampleRate)
        val recordedAt = MachineRegistry.nowIso()

        appendRow(pendingManifest, listOf(
            id, wav.absolutePath, machineId, engineType, category,
            provisionalTier, captureSource.name, deviceName(),
            sampleRate.toString(), durationOf(pcm, sampleRate), recordedAt,
        ))
        return PendingClip(id, wav.absolutePath, machineId, engineType, category, provisionalTier, captureSource.name, recordedAt)
    }

    fun listPending(machineId: String? = null): List<PendingClip> {
        if (!pendingManifest.exists()) return emptyList()
        val lines = pendingManifest.readLines().drop(1)
        return lines.mapNotNull { line ->
            val c = parseCsvLine(line)
            if (c.size < pendingHeader.size) return@mapNotNull null
            val clip = PendingClip(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[10])
            if (machineId != null && !clip.machineId.equals(machineId, ignoreCase = true)) null else clip
        }
    }

    fun pendingCount(machineId: String? = null): Int = listPending(machineId).size

    /**
     * Moves a pending clip into the confirmed corpus under its real label,
     * and forgets it was ever pending. Also settles the field benchmark for
     * this one recording — the on-device tier it queued with, against the
     * verdict it just received — and writes that outcome into the manifest
     * alongside the label, so the dataset itself carries where the model
     * agreed with the mechanic and where it didn't.
     *
     * Returns the outcome ("false_positive", "false_negative", or
     * "confirmed") so the caller can act on it — e.g. flag a machine whose
     * enrolled baseline just let a real fault read as healthy.
     */
    fun confirmPending(pending: PendingClip, corpusLabel: String, displayLabel: String, mechanicVerdict: String): String {
        val outcome = outcomeFor(pending.provisionalTier, corpusLabel)
        val src = File(pending.path)
        if (!src.exists()) { removePending(pending.pendingId); return outcome }
        val safeLabel = corpusLabel.replace(Regex("[^A-Za-z0-9_ -]"), "_")
        val dir = File(root, safeLabel)
        dir.mkdirs()
        val dest = File(dir, src.name)
        src.copyTo(dest, overwrite = true)
        src.delete()
        ensureHeader(manifest, manifestHeader)

        val info = readWavInfo(dest)
        appendRow(manifest, listOf(
            dest.absolutePath, corpusLabel, corpusLabel, safe(pending.machineId), "",
            displayLabel, mechanicVerdict, pending.captureSource,
            deviceName(), info.first.toString(), "%.2f".format(info.second),
            pending.recordedAtUtc, pending.engineType, pending.category,
            outcome,
        ))
        removePending(pending.pendingId)
        return outcome
    }

    /**
     * Predicted-anomalous is any tier past HEALTHY (WATCH counts — it already
     * told the user "worth checking"). Comparing that against whether the
     * confirmed label is the healthy corpus key gives the same false-
     * positive/false-negative framing as any other binary detector, which is
     * the point: this app is a field benchmark of the on-device model, not
     * just a data collector.
     */
    private fun outcomeFor(provisionalTier: String, corpusLabel: String): String {
        val predictedFaulty = provisionalTier != AnomalyScorer.Tier.HEALTHY.name
        val actualFaulty = corpusLabel != FaultLabel.HEALTHY_CORPUS_KEY
        return when {
            predictedFaulty && !actualFaulty -> "false_positive"
            !predictedFaulty && actualFaulty -> "false_negative"
            else -> "confirmed"
        }
    }

    /** Discards a pending clip with no label — a bad recording, or one nobody ever got a verdict for. */
    fun discardPending(pending: PendingClip) {
        File(pending.path).delete()
        removePending(pending.pendingId)
    }

    private fun removePending(pendingId: String) {
        if (!pendingManifest.exists()) return
        val lines = pendingManifest.readLines()
        val header = lines.firstOrNull() ?: return
        val kept = lines.drop(1).filterNot { parseCsvLine(it).firstOrNull() == pendingId }
        pendingManifest.writeText((listOf(header) + kept).joinToString("\n") + "\n")
    }

    // --------------------------------------------------------------- totals

    fun clipCount(): Int = if (!manifest.exists()) 0 else (manifest.readLines().size - 1).coerceAtLeast(0)

    /** healthy vs. faulty tally across every confirmed clip, plus the per-class breakdown behind "faulty" — the live progress read against "is this a good dataset yet." */
    data class LabelCounts(val healthy: Int, val byFault: Map<String, Int>) {
        val faulty: Int get() = byFault.values.sum()
        val total: Int get() = healthy + faulty
    }

    fun labelCounts(): LabelCounts {
        if (!manifest.exists()) return LabelCounts(0, emptyMap())
        var healthy = 0
        val faults = LinkedHashMap<String, Int>()
        manifest.readLines().drop(1).forEach { line ->
            val c = parseCsvLine(line)
            if (c.size < 2) return@forEach
            val label = c[1]
            if (label == FaultLabel.HEALTHY_CORPUS_KEY) healthy++
            else if (label.isNotBlank()) faults[label] = (faults[label] ?: 0) + 1
        }
        return LabelCounts(healthy, faults)
    }

    // --------------------------------------------------------------- helpers

    private fun safe(id: String): String = id.trim().ifEmpty { "unknown" }.replace(Regex("[^A-Za-z0-9_-]"), "_")

    private fun deviceName(): String = "${Build.MANUFACTURER} ${Build.MODEL}"

    private fun durationOf(pcm: FloatArray, sampleRate: Int): String = "%.2f".format(pcm.size.toDouble() / sampleRate)

    private fun ensureHeader(f: File, header: List<String>) {
        if (f.exists()) return
        f.parentFile?.mkdirs()
        f.writeText(header.joinToString(",") + "\n")
    }

    private fun appendRow(f: File, values: List<String>) {
        f.appendText(values.joinToString(",") { csvEscape(it) } + "\n")
    }

    private fun csvEscape(s: String): String =
        if (s.contains(',') || s.contains('"') || s.contains('\n')) "\"" + s.replace("\"", "\"\"") + "\"" else s

    /** Minimal CSV line parser: handles our own quoting, not a general RFC 4180 parser. */
    private fun parseCsvLine(line: String): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < line.length) {
            val c = line[i]
            when {
                inQuotes && c == '"' && i + 1 < line.length && line[i + 1] == '"' -> { cur.append('"'); i++ }
                c == '"' -> inQuotes = !inQuotes
                c == ',' && !inQuotes -> { out.add(cur.toString()); cur.clear() }
                else -> cur.append(c)
            }
            i++
        }
        out.add(cur.toString())
        return out
    }

    private fun readWavInfo(f: File): Pair<Int, Double> {
        RandomAccessFile(f, "r").use { raf ->
            raf.seek(24)
            val b = ByteArray(4); raf.read(b)
            val sr = (b[0].toInt() and 0xFF) or ((b[1].toInt() and 0xFF) shl 8) or
                ((b[2].toInt() and 0xFF) shl 16) or ((b[3].toInt() and 0xFF) shl 24)
            val dataBytes = raf.length() - 44
            return sr to (dataBytes / 2.0) / sr
        }
    }

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
