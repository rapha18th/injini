package com.injini.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.concurrent.thread

/**
 * Three flows, all against whichever machine is currently selected (see
 * [MachineRegistry], [machineRow]):
 *
 *   Enrol   record [ENROL_CLIPS] healthy clips, embed them, build an
 *           [AnomalyScorer], persist it under this machine's id.
 *   Check   record one clip, score it against the machine's healthy
 *           fingerprint, show a three-tier verdict.
 *   Add training clip   record one clip and save it, labelled, for the next
 *           supervised fault-classification retrain — see [LabeledClipStore]
 *           and [FaultLabel]. This is the "collect real data" half of the
 *           product, not a separate app the way the working paper once
 *           imagined it.
 *
 * Every recording, of any kind, runs through [recordWithFeedback], which
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

    private lateinit var machineRow: TextView
    private lateinit var statusWord: TextView
    private lateinit var detailText: TextView
    private lateinit var latencyChip: TextView
    private lateinit var waveform: WaveformView
    private lateinit var checkButton: Button
    private lateinit var enrolButton: Button
    private lateinit var addClipButton: Button
    private lateinit var resultsLink: TextView

    private lateinit var features: AudioFeatures
    private lateinit var registry: MachineRegistry
    private lateinit var clipStore: LabeledClipStore
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
        machineRow = findViewById(R.id.machineRow)
        statusWord = findViewById(R.id.statusWord)
        detailText = findViewById(R.id.detailText)
        latencyChip = findViewById(R.id.latencyChip)
        waveform = findViewById(R.id.waveform)
        checkButton = findViewById(R.id.checkButton)
        enrolButton = findViewById(R.id.enrolButton)
        addClipButton = findViewById(R.id.addClipButton)
        resultsLink = findViewById(R.id.resultsLink)

        features = AudioFeatures(this)
        registry = MachineRegistry(this)
        clipStore = LabeledClipStore(this)

        thread {
            val e = InjiniEmbedder(this, "injini_mn10_as_int8.onnx")
            e.warmUp()
            runOnUiThread { embedder = e; restoreLastMachine(); setIdle() }
        }

        machineRow.setOnClickListener { showMachinePicker() }
        checkButton.setOnClickListener { ensureMic { withMachine { runCheck() } } }
        enrolButton.setOnClickListener { ensureMic { withMachine { runEnrol() } } }
        addClipButton.setOnClickListener { ensureMic { withMachine { showLabelDialog() } } }
        resultsLink.setOnClickListener { showResults() }
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

    /** Runs [then] if a machine is already selected, otherwise opens the picker first. */
    private fun withMachine(then: () -> Unit) {
        if (machineId != null) then() else showMachinePicker(onSelected = then)
    }

    // ---------------------------------------------------------------- machines

    private fun restoreLastMachine() {
        val last = getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_LAST_MACHINE, null)
        if (last != null && registry.find(last) != null) selectMachine(last)
    }

    private fun selectMachine(id: String) {
        machineId = id
        val m = registry.find(id)
        scorer = m?.scorerJson?.let { runCatching { AnomalyScorer.fromJson(it) }.getOrNull() }
        machineRow.text = "Machine: $id  ›"
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_LAST_MACHINE, id).apply()
        setIdle()
    }

    /** Lists every known machine plus a field to name a new one. [onSelected] fires after a pick, if given. */
    private fun showMachinePicker(onSelected: (() -> Unit)? = null) {
        val view = LayoutInflater.from(this).inflate(R.layout.dialog_machine_picker, null)
        val nameInput = view.findViewById<EditText>(R.id.machineNameInput)
        val existingList = view.findViewById<LinearLayout>(R.id.existingList)
        val existingLabel = view.findViewById<TextView>(R.id.existingLabel)
        nameInput.setText(machineId ?: "")

        val known = registry.knownIds()
        existingLabel.visibility = if (known.isEmpty()) View.GONE else View.VISIBLE
        var dialog: AlertDialog? = null
        for (id in known) {
            val row = TextView(this).apply {
                text = "  $id"
                textSize = 15f
                setPadding(4, 20, 4, 20)
                setOnClickListener {
                    nameInput.setText(id)
                    confirmMachine(id, onSelected)
                    dialog?.dismiss()
                }
            }
            existingList.addView(row)
        }

        dialog = AlertDialog.Builder(this)
            .setTitle("Choose a machine")
            .setView(view)
            .setPositiveButton("Use this name") { _, _ ->
                val id = nameInput.text.toString().trim()
                if (id.isNotEmpty()) confirmMachine(id, onSelected)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun confirmMachine(id: String, onSelected: (() -> Unit)?) {
        registry.ensureExists(id, MachineRegistry.nowIso())
        selectMachine(id)
        onSelected?.invoke()
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
            waveform.reset()
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
            runOnUiThread { waveform.visibility = View.GONE }
        }
    }

    // -------------------------------------------------------------------- idle

    /** Re-enable the buttons after a background op, without touching whatever is on screen. */
    private fun refreshButtons() {
        busy = false
        val hasMachine = machineId != null
        checkButton.isEnabled = embedder != null && hasMachine && scorer != null
        enrolButton.isEnabled = embedder != null
        addClipButton.isEnabled = embedder != null
        checkButton.alpha = if (checkButton.isEnabled) 1f else 0.4f
        enrolButton.alpha = if (enrolButton.isEnabled) 1f else 0.4f
        addClipButton.alpha = if (addClipButton.isEnabled) 1f else 0.4f
    }

    private fun setIdle() {
        refreshButtons()
        when {
            machineId == null ->
                status("PICK A MACHINE", "Tap the machine row above to select or add one, then Enrol or add a training clip.")
            scorer == null ->
                status("NOT ENROLLED", "Enrol $machineId first: $ENROL_CLIPS ten-second clips while it runs normally.")
            else ->
                status("READY", "Enrolled: $machineId. Hold the phone near it and Check.")
        }
    }

    // ------------------------------------------------------------------ enrol

    private fun runEnrol() {
        val e = embedder ?: return
        val id = machineId ?: return
        if (busy) return
        busy = true
        checkButton.isEnabled = false; enrolButton.isEnabled = false; addClipButton.isEnabled = false
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
                }
                val s = AnomalyScorer.enroll(healthy)
                registry.saveScorer(id, s.toJson())
                runOnUiThread { scorer = s; setIdle() }
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
        if (busy) return
        busy = true
        checkButton.isEnabled = false; enrolButton.isEnabled = false; addClipButton.isEnabled = false
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
                    refreshButtons()
                }
            } catch (ex: Exception) {
                runOnUiThread { status("CHECK FAILED", ex.message ?: "unknown error"); setIdle() }
            }
        }
    }

    // ---------------------------------------------------------- training clips

    /** Healthy-after-service or faulty-with-a-mechanic's-verdict, for the fault-ID retrain. */
    private fun showLabelDialog() {
        val id = machineId ?: return
        val view = LayoutInflater.from(this).inflate(R.layout.dialog_label_clip, null)
        val group = view.findViewById<RadioGroup>(R.id.labelKindGroup)
        val radioFaulty = view.findViewById<RadioButton>(R.id.radioFaulty)
        val spinner = view.findViewById<Spinner>(R.id.faultSpinner)
        val verdictInput = view.findViewById<EditText>(R.id.verdictInput)

        spinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, FaultLabel.faultyDisplayOptions())
        group.setOnCheckedChangeListener { _, _ ->
            val faulty = radioFaulty.isChecked
            spinner.visibility = if (faulty) View.VISIBLE else View.GONE
            verdictInput.visibility = if (faulty) View.VISIBLE else View.GONE
        }

        AlertDialog.Builder(this)
            .setTitle("Add a training clip for $id")
            .setView(view)
            .setPositiveButton("Record 10 s") { _, _ ->
                val faulty = radioFaulty.isChecked
                val display = if (faulty) spinner.selectedItem as? String ?: FaultLabel.OTHER_DISPLAY else FaultLabel.HEALTHY_DISPLAY
                val verdict = if (faulty) verdictInput.text.toString() else ""
                runAddTrainingClip(id, display, verdict, healthy = !faulty)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun runAddTrainingClip(id: String, displayLabel: String, mechanicVerdict: String, healthy: Boolean) {
        val e = embedder ?: return
        if (busy) return
        busy = true
        checkButton.isEnabled = false; enrolButton.isEnabled = false; addClipButton.isEnabled = false
        thread {
            try {
                val corpusLabel = if (healthy) FaultLabel.HEALTHY_CORPUS_KEY
                    else FaultLabel.corpusKeyFor(displayLabel, mechanicVerdict)
                val what = if (healthy) "healthy sound" else "the fault: $displayLabel"
                val rec = recordWithFeedback(
                    CLIP_SECONDS, "COLLECTING TRAINING CLIP",
                    "Recording $what from $id for the next retrain.",
                )
                clipStore.save(
                    pcm = rec.pcm, sampleRate = rec.sampleRate, machineId = id,
                    corpusLabel = corpusLabel, displayLabel = displayLabel,
                    mechanicVerdict = mechanicVerdict, captureSource = rec.source,
                )
                registry.recordLabeledClip(id, corpusLabel, MachineRegistry.nowIso())
                val total = clipStore.clipCount()
                val forMachine = registry.totalLabeledClips(id)
                runOnUiThread {
                    status("CLIP SAVED", "Labelled \"$corpusLabel\" for $id.\n\n" +
                        "$forMachine training ${if (forMachine == 1) "clip" else "clips"} collected for this machine, " +
                        "$total total on this phone.")
                    refreshButtons()
                }
            } catch (ex: Exception) {
                runOnUiThread { status("SAVE FAILED", ex.message ?: "unknown error"); setIdle() }
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
                appendLine("Training clips collected on this phone: ${clipStore.clipCount()}")
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
