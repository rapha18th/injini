package com.injini.app

import android.app.Activity
import android.view.LayoutInflater
import android.view.View
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Spinner
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

    fun showQueue(machineId: String) {
        val pending = clipStore.listPending(machineId)
        if (pending.isEmpty()) return
        val names = pending.map {
            "Recorded ${it.recordedAtUtc.take(16).replace('T', ' ')}  —  model said: ${it.provisionalTier.lowercase()}"
        }
        AlertDialog.Builder(activity)
            .setTitle("$machineId — recordings awaiting a verdict")
            .setItems(names.toTypedArray()) { _, which -> showLabelDialog(pending[which]) }
            .setNegativeButton("Close", null)
            .show()
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
                clipStore.confirmPending(pending, corpusLabel, display, verdict)
                registry.recordLabeledClip(pending.machineId, corpusLabel)
                onChanged()
            }
            .setNeutralButton("Discard recording") { _, _ ->
                clipStore.discardPending(pending)
                onChanged()
            }
            .setNegativeButton("Later", null)
            .show()
    }
}
