package com.injini.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlin.concurrent.thread

/**
 * Two flows:
 *
 *   Enrol   record [ENROL_CLIPS] healthy clips of one machine, embed them,
 *           build an [AnomalyScorer], persist it. Do this once per machine
 *           while it is running well.
 *   Check   record one clip, embed it, score it against the enrolled healthy
 *           fingerprint, show a three-tier verdict.
 *
 * The embedder is the frozen INT8 EfficientAT graph (assets/injini_*_int8.onnx).
 * The decision is [AnomalyScorer], pure Kotlin, no ONNX. "Full model results"
 * shows the matched FP32-vs-INT8 embedder benchmark and the execution-provider
 * trace, same methodology as SiloSense.
 */
class MainActivity : AppCompatActivity() {

    private companion object {
        const val ENROL_CLIPS = 6
        const val CLIP_SECONDS = 10.0
        const val PREFS = "injini"
        const val KEY_SCORER = "scorer_json"
        const val KEY_MACHINE = "machine_name"
    }

    private val askMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) setIdle() else status("MIC DENIED", "Injini needs the microphone to listen to the machine.")
    }

    private lateinit var statusWord: TextView
    private lateinit var detailText: TextView
    private lateinit var latencyChip: TextView
    private lateinit var checkButton: Button
    private lateinit var enrolButton: Button
    private lateinit var resultsLink: TextView

    private lateinit var features: AudioFeatures
    private var embedder: InjiniEmbedder? = null
    private var fp32Embedder: InjiniEmbedder? = null
    private val capture = AudioCapture()

    private var scorer: AnomalyScorer? = null
    private var machineName: String = "machine"
    private var busy = false

    private var int8Bench: InjiniEmbedder.BenchmarkResult? = null
    private var fp32Bench: InjiniEmbedder.BenchmarkResult? = null
    private var providerTrace: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        statusWord = findViewById(R.id.statusWord)
        detailText = findViewById(R.id.detailText)
        latencyChip = findViewById(R.id.latencyChip)
        checkButton = findViewById(R.id.checkButton)
        enrolButton = findViewById(R.id.enrolButton)
        resultsLink = findViewById(R.id.resultsLink)

        features = AudioFeatures(this)

        thread {
            val e = InjiniEmbedder(this, "injini_mn10_as_int8.onnx")
            e.warmUp()
            runOnUiThread { embedder = e; restoreScorer(); setIdle() }
        }

        checkButton.setOnClickListener { ensureMic { runCheck() } }
        enrolButton.setOnClickListener { ensureMic { runEnrol() } }
        resultsLink.setOnClickListener { showResults() }
    }

    private fun ensureMic(then: () -> Unit) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED) then() else askMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun restoreScorer() {
        val p = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val json = p.getString(KEY_SCORER, null) ?: return
        machineName = p.getString(KEY_MACHINE, "machine") ?: "machine"
        scorer = runCatching { AnomalyScorer.fromJson(json) }.getOrNull()
    }

    private fun setIdle() {
        busy = false
        checkButton.isEnabled = embedder != null && scorer != null
        enrolButton.isEnabled = embedder != null
        checkButton.alpha = if (checkButton.isEnabled) 1f else 0.4f
        enrolButton.alpha = if (enrolButton.isEnabled) 1f else 0.4f
        if (scorer == null) {
            status("NOT ENROLLED", "Enrol this machine first: $ENROL_CLIPS ten-second clips while it runs normally.")
        } else {
            status("READY", "Enrolled: $machineName. Hold the phone near it and Check.")
        }
    }

    private fun runEnrol() {
        val e = embedder ?: return
        if (busy) return
        busy = true
        checkButton.isEnabled = false; enrolButton.isEnabled = false
        thread {
            val healthy = ArrayList<FloatArray>(ENROL_CLIPS)
            try {
                for (i in 1..ENROL_CLIPS) {
                    runOnUiThread { status("ENROLLING $i/$ENROL_CLIPS", "Recording healthy sound. Keep the phone still.") }
                    val rec = capture.record(CLIP_SECONDS)
                    val mel = features.logMelFromPcm(rec.pcm, rec.sampleRate)
                    healthy += e.embed(mel)
                }
                val s = AnomalyScorer.enroll(healthy)
                getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                    .putString(KEY_SCORER, s.toJson())
                    .putString(KEY_MACHINE, machineName)
                    .apply()
                runOnUiThread { scorer = s; setIdle() }
            } catch (ex: Exception) {
                runOnUiThread { status("ENROL FAILED", ex.message ?: "unknown error"); setIdle() }
            }
        }
    }

    private fun runCheck() {
        val e = embedder ?: return
        val s = scorer ?: return
        if (busy) return
        busy = true
        checkButton.isEnabled = false; enrolButton.isEnabled = false
        status("LISTENING", "Recording ${CLIP_SECONDS.toInt()} s. Keep the phone still, clear of belts and fans.")
        thread {
            try {
                val rec = capture.record(CLIP_SECONDS)
                val mel = features.logMelFromPcm(rec.pcm, rec.sampleRate)
                val (emb, ms) = e.timedEmbed(mel)
                val r = s.score(emb)
                runOnUiThread {
                    latencyChip.text = "embed ${"%.1f".format(ms)} ms"
                    val (word, line) = when (r.tier) {
                        AnomalyScorer.Tier.HEALTHY -> "SOUNDS LIKE ITSELF" to
                            "Close to the healthy fingerprint. Nothing to act on."
                        AnomalyScorer.Tier.WATCH -> "WORTH CHECKING" to
                            "Drifting from healthy. Listen again in a day, and have a mechanic look if it grows."
                        AnomalyScorer.Tier.CHECK -> "GET IT LOOKED AT" to
                            "Clearly unlike the healthy fingerprint. A mechanic should check it."
                    }
                    val src = if (rec.source != AudioCapture.Source.UNPROCESSED)
                        "\n\nCapture source: ${rec.source} (unprocessed audio unavailable on this device; the reading is less reliable)."
                    else ""
                    status(word, "$line\n\nscore ${"%.3f".format(r.score)}  " +
                        "(kNN ${"%.3f".format(r.knnDistance)}, Mahalanobis ${"%.3f".format(r.mahalanobis)})$src")
                    setIdle()
                }
            } catch (ex: Exception) {
                runOnUiThread { status("CHECK FAILED", ex.message ?: "unknown error"); setIdle() }
            }
        }
    }

    private fun showResults() {
        val e = embedder ?: return
        thread {
            val dummyMel = features.logMel(FloatArray(AudioFeatures.CLIP_SAMPLES))
            if (providerTrace == null) providerTrace = e.diagnoseExecutionProviders(dummyMel)
            if (int8Bench == null) int8Bench = e.benchmark(dummyMel)
            if (fp32Embedder == null) fp32Embedder = InjiniEmbedder(this, "injini_mn10_as_fp32.onnx").also { it.warmUp() }
            if (fp32Bench == null) fp32Bench = fp32Embedder!!.benchmark(dummyMel)
            val text = buildString {
                appendLine("Embedder: EfficientAT mn10_as, frozen, ${e.embedDim}-d")
                appendLine("Configured providers: ${e.configuredProviders}")
                appendLine("Executed per node: ${providerTrace}")
                appendLine()
                appendLine("Steady-state embed (50-run avg style, 20 timed):")
                appendLine("  INT8  ${"%.2f".format(int8Bench!!.avgMs)} ms   (min ${"%.2f".format(int8Bench!!.minMs)})")
                appendLine("  FP32  ${"%.2f".format(fp32Bench!!.avgMs)} ms   (min ${"%.2f".format(fp32Bench!!.minMs)})")
                appendLine("  load  INT8 ${"%.1f".format(e.loadTimeMs)} ms   FP32 ${"%.1f".format(fp32Embedder!!.loadTimeMs)} ms")
                appendLine()
                appendLine("Process memory (PSS): ${DeviceDiagnostics.pssKb()} KB")
                appendLine("Thermal status: ${DeviceDiagnostics.thermalStatus(this@MainActivity)}")
            }
            runOnUiThread {
                AlertDialog.Builder(this).setTitle("Full model results").setMessage(text)
                    .setPositiveButton("OK", null).show()
            }
        }
    }

    private fun status(word: String, detail: String) {
        statusWord.text = word
        detailText.text = detail
    }

    override fun onDestroy() {
        super.onDestroy()
        embedder?.close(); fp32Embedder?.close()
    }
}
