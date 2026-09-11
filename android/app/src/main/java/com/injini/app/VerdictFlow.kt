package com.injini.app

import android.app.Activity
import android.view.LayoutInflater
import android.view.View
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AlertDialog

/**
 * "What turned out to be true?" — reviewing a machine's recordings that are
 * still waiting for a real label and attaching one. Shared between
 * [MainActivity] (the home-screen "Add verdict" button) and
 * [MachineListActivity] (the same action from the fleet screen), so this is
 * the one place the pending-review behaviour is defined.
 *
 * This is reached only by an explicit "Add verdict" tap — never by hijacking
 * a machine-selection tap — so there is nothing here to "decline"; closing
 * the queue just closes it.
 */
class VerdictFlow(
    private val activity: Activity,
    private val registry: MachineRegistry,
    private val clipStore: LabeledClipStore,
    private val onChanged: () -> Unit,
) {

    /**
     * A plain AlertDialog list here used to render as inert-looking text —
     * no visual cue that a row was the only way into the label dialog. Each
     * row is now a chip with a ripple and a trailing chevron, the same
     * "control, not text" fix already applied to the fleet screen's rows.
     */
    fun showQueue(machineId: String) {
        val pending = clipStore.listPending(machineId)
        if (pending.isEmpty()) return
        val view = LayoutInflater.from(activity).inflate(R.layout.dialog_pending_queue, null)
        val container = view.findViewById<LinearLayout>(R.id.pendingList)
        val dialog = AlertDialog.Builder(activity)
            .setTitle("$machineId — recordings awaiting a verdict")
            .setView(view)
            .setNegativeButton("Close", null)
            .create()
        pending.forEach { clip ->
            val row = LayoutInflater.from(activity).inflate(R.layout.item_pending_row, container, false)
            row.findViewById<TextView>(R.id.pendingTimestamp).text =
                "Recorded ${clip.recordedAtUtc.take(16).replace('T', ' ')}"
            row.findViewById<TextView>(R.id.pendingTier).text =
                "model said: ${clip.provisionalTier.lowercase()}"
            row.setOnClickListener { dialog.dismiss(); showLabelDialog(clip) }
            container.addView(row)
        }
        dialog.show()
    }

    private fun showLabelDialog(pending: LabeledClipStore.PendingClip) {
        val view = LayoutInflater.from(activity).inflate(R.layout.dialog_label_clip, null)
        val group = view.findViewById<RadioGroup>(R.id.labelKindGroup)
        val radioFaulty = view.findViewById<RadioButton>(R.id.radioFaulty)
        val spinner = view.findViewById<Spinner>(R.id.faultSpinner)
        val verdictInput = view.findViewById<EditText>(R.id.verdictInput)
        spinner.adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_dropdown_item, FaultLabel.faultyDisplayOptions())
        group.setOnCheckedChangeListener { _, _ ->
            val faulty = radioFaulty.isChecked
            spinner.visibility = if (faulty) View.VISIBLE else View.GONE
            verdictInput.visibility = if (faulty) View.VISIBLE else View.GONE
        }

        AlertDialog.Builder(activity)
            .setTitle("What turned out to be true?")
            .setMessage("This recording's model guess was \"${pending.provisionalTier.lowercase()}\" — say what the mechanic actually found.")
            .setView(view)
            .setPositiveButton("Save label") { _, _ ->
                val faulty = radioFaulty.isChecked
                val display = if (faulty) spinner.selectedItem as? String ?: FaultLabel.OTHER_DISPLAY else FaultLabel.HEALTHY_DISPLAY
                val verdict = if (faulty) verdictInput.text.toString() else ""
                val corpusLabel = if (faulty) FaultLabel.corpusKeyFor(display, verdict) else FaultLabel.HEALTHY_CORPUS_KEY
                val outcome = clipStore.confirmPending(pending, corpusLabel, display, verdict)
                registry.recordLabeledClip(pending.machineId, corpusLabel)
                registry.recordBenchmarkOutcome(pending.machineId, outcome)
                reportOutcome(pending.machineId, outcome)
                onChanged()
            }
            .setNeutralButton("Discard recording") { _, _ ->
                clipStore.discardPending(pending)
                onChanged()
            }
            .setNegativeButton("Later", null)
            .show()
    }

    /**
     * The model's tier and the mechanic's verdict just disagreed, or they
     * didn't. Either way that's a real benchmark result, not only a training
     * label — surface it rather than letting it sit silently in a CSV
     * column. A false negative (Check read HEALTHY, something was actually
     * wrong) also marks the machine for re-enrollment: the fingerprint that
     * produced that reading may itself include a bad sample.
     */
    private fun reportOutcome(machineId: String, outcome: String) {
        when (outcome) {
            "false_positive" -> AlertDialog.Builder(activity)
                .setTitle("False positive recorded")
                .setMessage("$machineId's model flagged this one for a check, and the mechanic found it healthy. Logged for the benchmark — no action needed.")
                .setPositiveButton("OK", null)
                .show()
            "false_negative" -> {
                registry.flagReenrollment(machineId)
                AlertDialog.Builder(activity)
                    .setTitle("Missed fault recorded")
                    .setMessage("$machineId's model read this one as healthy, but the mechanic found a real fault. Logged as a missed detection, and $machineId is now marked for re-enrollment — its healthy fingerprint may include a bad sample.")
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }
}
