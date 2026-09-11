package com.injini.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatImageButton
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.concurrent.thread

/**
 * Two flows against whichever machine is currently selected (see
 * [MachineRegistry] and [MachineListActivity]):
 *
 *   Enrol   record [ENROL_CLIPS] healthy clips, embed them to build this
 *           machine's [AnomalyScorer], AND save every one of them straight
 *           into the training corpus under the healthy label — an enrolled
 *           clip is healthy by construction, so there is nothing left to ask
 *           the user about it. See [LabeledClipStore.save].
 *   Check   record one clip, score it against the machine's fingerprint,
 *           show a three-tier verdict, and save the clip with that
 *           provisional tier as a hint. What Check does NOT know is the real
 *           answer — that comes from a mechanic, later — so the clip queues
 *           in [LabeledClipStore.savePending] until "Add verdict" is used,
 *           right here or from the fleet screen, to say what actually
 *           turned out to be true.
 *
 * Every recording, of either kind, runs through [recordWithFeedback], which
 * drives a live [WaveformView] and a counting-down status line so a
 * ten-second clip never looks like a stuck screen.
 */
class MainActivity : AppCompatActivity() {

    private companion object {
        const val ENROL_CLIPS = 6
        const val CLIP_SECONDS = 10.0
        const val PREFS = "injini"
        const val KEY_LAST_MACHINE = "last_machine_id"
    }

    private val askMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) setIdle() else status("MIC DENIED", "Injini needs the microphone to listen to the machine.")
    }

    private val pickMachine = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val id = result.data?.getStringExtra(MachineListActivity.EXTRA_SELECTED_ID)
        if (result.resultCode == RESULT_OK && id != null) selectMachine(id)
    }

    private lateinit var machinePicker: TextView
    private lateinit var machinesButton: AppCompatImageButton
    private lateinit var machineInfo: TextView
    private lateinit var reenrollBanner: TextView
    private lateinit var addVerdictButton: Button
    private lateinit var statusWord: TextView
    private lateinit var detailText: TextView
    private lateinit var latencyChip: TextView
    private lateinit var waveform: WaveformView
    private lateinit var checkButton: Button
    private lateinit var enrolButton: Button
    private lateinit var resultsLink: TextView

    private lateinit var features: AudioFeatures
    private lateinit var registry: MachineRegistry
    private lateinit var clipStore: LabeledClipStore
    private lateinit var verdictFlow: VerdictFlow
    private var embedder: InjiniEmbedder? = null
    private var fp32Embedder: InjiniEmbedder? = null
    private val capture = AudioCapture()

    private var scorer: AnomalyScorer? = null
    private var machineId: String? = null
    private var busy = false

    private var int8Bench: InjiniEmbedder.BenchmarkResult? = null
    private var fp32Bench: InjiniEmbedder.BenchmarkResult? = null
    private var providerTrace: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        applySystemBarInsetsAsExtraPadding()
        machinePicker = findViewById(R.id.machinePicker)
        machinesButton = findViewById(R.id.machinesButton)
        machineInfo = findViewById(R.id.machineInfo)
        reenrollBanner = findViewById(R.id.reenrollBanner)
        addVerdictButton = findViewById(R.id.addVerdictButton)
        statusWord = findViewById(R.id.statusWord)
        detailText = findViewById(R.id.detailText)
        latencyChip = findViewById(R.id.latencyChip)
        waveform = findViewById(R.id.waveform)
        checkButton = findViewById(R.id.checkButton)
        enrolButton = findViewById(R.id.enrolButton)
        resultsLink = findViewById(R.id.resultsLink)

        features = AudioFeatures(this)
        registry = MachineRegistry(this)
        clipStore = LabeledClipStore(this)
        verdictFlow = VerdictFlow(this, registry, clipStore, onChanged = { machineId?.let { refreshMachineInfo(it) } })

        thread {
            val e = InjiniEmbedder(this, "injini_mn10_as_int8.onnx")
            e.warmUp()
            runOnUiThread { embedder = e; restoreLastMachine(); setIdle() }
        }

        machinePicker.setOnClickListener { showMachinePickerDropdown() }
        machinesButton.setOnClickListener { pickMachine.launch(Intent(this, MachineListActivity::class.java)) }
        addVerdictButton.setOnClickListener { machineId?.let { verdictFlow.showQueue(it) } }
        checkButton.setOnClickListener { ensureMic { withMachine { runCheck() } } }
        enrolButton.setOnClickListener { ensureMic { withMachine { runEnrol() } } }
        resultsLink.setOnClickListener { showResults() }
    }

    override fun onResume() {
        super.onResume()
        // Attributes or pending counts may have changed in the machine list.
        machineId?.let { refreshMachineInfo(it) }
    }

    /**
     * Edge-to-edge devices draw the window full-screen with the nav bar as a
     * transparent overlay on top, which silently eats touches on anything
     * laid out underneath it — see the note on the root layout in
     * activity_main.xml. Adds the system bar insets on top of the layout's
     * own 28dp padding (captured once here) rather than replacing it, which
     * is what android:fitsSystemWindows="true" does instead.
     */
    private fun applySystemBarInsetsAsExtraPadding() {
        val root = findViewById<View>(R.id.rootLayout)
        val baseLeft = root.paddingLeft
        val baseTop = root.paddingTop
        val baseRight = root.paddingRight
        val baseBottom = root.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(baseLeft + bars.left, baseTop + bars.top, baseRight + bars.right, baseBottom + bars.bottom)
            insets
        }
    }

    private fun ensureMic(then: () -> Unit) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED) then() else askMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    /** Runs [then] if a machine is already selected, otherwise opens the dropdown to pick or add one. */
    private fun withMachine(then: () -> Unit) {
        if (machineId != null) then() else showMachinePickerDropdown()
    }

    // ---------------------------------------------------------------- machines

    private fun restoreLastMachine() {
        val last = getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_LAST_MACHINE, null)
        if (last != null && registry.find(last) != null) selectMachine(last)
    }

    private fun selectMachine(id: String) {
        machineId = id
        scorer = registry.find(id)?.scorerJson?.let { runCatching { AnomalyScorer.fromJson(it) }.getOrNull() }
        refreshMachineInfo(id)
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_LAST_MACHINE, id).apply()
        setIdle()
    }

    /** Updates the picker label, the inline attributes/progress card, the re-enrollment flag, and the Add verdict button — all from the registry, every time something might have changed. */
    private fun refreshMachineInfo(id: String) {
        val m = registry.find(id) ?: return
        machinePicker.text = "Machine: $id  ⌄"

        val enrolled = m.scorerJson != null
        val total = registry.totalLabeledClips(id)
        machineInfo.visibility = View.VISIBLE
        machineInfo.text = "${m.engineType} · ${m.category}\n" +
            "${if (enrolled) "Enrolled" else "Not enrolled"}  ·  ${m.checkCount} checks  ·  $total training clips"

        reenrollBanner.visibility = if (m.needsReenrollment) View.VISIBLE else View.GONE

        val pending = clipStore.pendingCount(id)
        if (pending > 0) {
            addVerdictButton.visibility = View.VISIBLE
            addVerdictButton.text = "$pending recording${if (pending == 1) "" else "s"} — Add verdict"
        } else {
            addVerdictButton.visibility = View.GONE
        }
    }

    /** The "dropdown": every known machine, then "+ Add new machine" — styled as a picker even though it is an AlertDialog under the hood, to stay visually consistent with the rest of the app rather than fighting a stock Spinner's chrome. */
    private fun showMachinePickerDropdown() {
        val known = registry.knownIds()
        val options = known + "+ Add new machine"
        AlertDialog.Builder(this)
            .setTitle("Machine")
            .setItems(options.toTypedArray()) { _, which ->
                if (which == known.size) {
                    MachineForms.showAddMachineDialog(this, registry) { id -> selectMachine(id) }
                } else {
                    selectMachine(known[which])
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ------------------------------------------------------------ recording UX

    /**
     * Records [seconds] of audio while driving the waveform and a live
     * countdown, and always hides the waveform again afterwards. [title] and
     * [detail] are the status shown while it runs, e.g. "ENROLLING 2/6".
     */
    private fun recordWithFeedback(seconds: Double, title: String, detail: String): AudioCapture.Recording {
        runOnUiThread {
            waveform.visibility = View.VISIBLE
            waveform.start()
            status(title, detail)
        }
        try {
            return capture.record(seconds) { level, elapsedFraction ->
                runOnUiThread {
                    waveform.pushLevel(level)
                    val remaining = (seconds * (1.0 - elapsedFraction)).coerceAtLeast(0.0)
                    detailText.text = "$detail\n\nRecording — please wait, ${"%.0f".format(remaining)} s remaining."
                }
            }
        } finally {
            runOnUiThread { waveform.stop(); waveform.visibility = View.GONE }
        }
    }

    // -------------------------------------------------------------------- idle

    /** Re-enable the buttons after a background op, without touching whatever is on screen. */
    private fun refreshButtons() {
        busy = false
        val hasMachine = machineId != null
        checkButton.isEnabled = embedder != null && hasMachine && scorer != null
        enrolButton.isEnabled = embedder != null
        checkButton.alpha = if (checkButton.isEnabled) 1f else 0.4f
        enrolButton.alpha = if (enrolButton.isEnabled) 1f else 0.4f
    }

    private fun setIdle() {
        refreshButtons()
        when {
            machineId == null -> {
                machineInfo.visibility = View.GONE
                reenrollBanner.visibility = View.GONE
                addVerdictButton.visibility = View.GONE
                status("PICK A MACHINE", "Tap the machine row above to select or add one.")
            }
            scorer == null ->
                status("NOT ENROLLED", "Enrol $machineId first: $ENROL_CLIPS ten-second clips while it runs normally. Every clip also joins the training corpus as healthy.")
            else ->
                status("READY", "Enrolled: $machineId. Hold the phone near it and Check.")
        }
    }

    // ------------------------------------------------------------------ enrol

    private fun runEnrol() {
        val e = embedder ?: return
        val id = machineId ?: return
        val m = registry.find(id) ?: return
        if (busy) return
        busy = true
        checkButton.isEnabled = false; enrolButton.isEnabled = false
        thread {
            val healthy = ArrayList<FloatArray>(ENROL_CLIPS)
            try {
                for (i in 1..ENROL_CLIPS) {
                    val rec = recordWithFeedback(
                        CLIP_SECONDS, "ENROLLING $i/$ENROL_CLIPS",
                        "Recording healthy sound from $id. Keep the phone still.",
                    )
                    val mel = features.logMelFromPcm(rec.pcm, rec.sampleRate)
                    healthy += e.embed(mel)
                    // Healthy by construction: every enrol clip joins the training
                    // corpus immediately, no extra button and no extra decision.
                    clipStore.save(
                        pcm = rec.pcm, sampleRate = rec.sampleRate, machineId = id,
                        engineType = m.engineType, category = m.category,
                        corpusLabel = FaultLabel.HEALTHY_CORPUS_KEY, displayLabel = "Healthy (enrolment)",
                        mechanicVerdict = "", captureSource = rec.source,
                    )
                    registry.recordLabeledClip(id, FaultLabel.HEALTHY_CORPUS_KEY)
                }
                val s = AnomalyScorer.enroll(healthy)
                registry.saveScorer(id, s.toJson())
                runOnUiThread { scorer = s; refreshMachineInfo(id); setIdle() }
            } catch (ex: Exception) {
                runOnUiThread { status("ENROL FAILED", ex.message ?: "unknown error"); setIdle() }
            }
        }
    }

    // ------------------------------------------------------------------ check

    private fun runCheck() {
        val e = embedder ?: return
        val s = scorer ?: return
        val id = machineId ?: return
        val m = registry.find(id) ?: return
        if (busy) return
        busy = true
        checkButton.isEnabled = false; enrolButton.isEnabled = false
        thread {
            try {
                val rec = recordWithFeedback(
                    CLIP_SECONDS, "LISTENING",
                    "Recording $id. Keep the phone still, clear of belts and fans.",
                )
                val mel = features.logMelFromPcm(rec.pcm, rec.sampleRate)
                val (emb, ms) = e.timedEmbed(mel)
                val r = s.score(emb)
                registry.recordCheck(id, r.tier.name, MachineRegistry.nowIso())
                // What Check knows right now is a model guess, not a verdict. The
                // clip is saved and queued; the real label — healthy confirmed, or
                // a mechanic's actual finding — comes later via Add verdict.
                clipStore.savePending(
                    pcm = rec.pcm, sampleRate = rec.sampleRate, machineId = id,
                    engineType = m.engineType, category = m.category,
                    provisionalTier = r.tier.name, captureSource = rec.source,
                )
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
                        "(kNN ${"%.3f".format(r.knnDistance)}, Mahalanobis ${"%.3f".format(r.mahalanobis)})$src\n\n" +
                        "Saved. Use Add verdict above once you know what this really was.")
                    refreshMachineInfo(id)
                    refreshButtons()
                }
            } catch (ex: Exception) {
                runOnUiThread { status("CHECK FAILED", ex.message ?: "unknown error"); setIdle() }
            }
        }
    }

    // ------------------------------------------------------------------- misc

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
                appendLine()
                appendLine("Machines on this phone: ${registry.all().size}")
                appendLine("Training clips collected: ${clipStore.clipCount()}")
                appendLine("Recordings awaiting a verdict: ${clipStore.pendingCount()}")
                appendLine()
                appendLine("Field benchmark (on-device tier vs. mechanic verdict):")
                appendLine("  False positives (flagged, turned out healthy): ${registry.totalFalsePositives()}")
                appendLine("  Missed faults (read healthy, turned out faulty): ${registry.totalFalseNegatives()}")
                appendLine("  Machines needing re-enrollment: ${registry.countNeedingReenrollment()}")
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
