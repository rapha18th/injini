package com.injini.app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/**
 * The fleet screen. A kombi rank or a generator dealer has many machines on
 * one phone, each with its own attributes and its own contribution to the
 * training corpus — this is where that is managed, not a name typed into a
 * one-line dialog on the main screen.
 *
 * Tapping a machine selects it and returns to [MainActivity]
 * ([Activity.RESULT_OK] with [EXTRA_SELECTED_ID]). Long-pressing a row
 * offers to delete that machine's registry entry (its recordings on disk are
 * untouched, since they already belong to the corpus regardless of what
 * happens to the live scorer).
 */
class MachineListActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_SELECTED_ID = "selected_machine_id"
    }

    private lateinit var registry: MachineRegistry
    private lateinit var clipStore: LabeledClipStore
    private lateinit var listContainer: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_machine_list)
        applyInsets()
        registry = MachineRegistry(this)
        clipStore = LabeledClipStore(this)
        listContainer = findViewById(R.id.machineList)
        findViewById<View>(R.id.addMachineButton).setOnClickListener { showAddMachineDialog() }
        refreshList()
    }

    override fun onResume() {
        super.onResume()
        refreshList()
    }

    private fun applyInsets() {
        val root = findViewById<View>(R.id.listRoot)
        val l = root.paddingLeft; val t = root.paddingTop; val r = root.paddingRight; val b = root.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(l + bars.left, t + bars.top, r + bars.right, b + bars.bottom)
            insets
        }
    }

    private fun refreshList() {
        listContainer.removeAllViews()
        val machines = registry.all()
        if (machines.isEmpty()) {
            val empty = TextView(this).apply {
                text = "No machines yet. Add one to enrol it or start collecting training clips."
                setTextColor(getColor(R.color.injini_dim))
                textSize = 14f
            }
            listContainer.addView(empty)
            return
        }
        for (m in machines) {
            val row = LayoutInflater.from(this).inflate(R.layout.item_machine_row, listContainer, false)
            row.findViewById<TextView>(R.id.rowName).text = m.id
            row.findViewById<TextView>(R.id.rowType).text = "${m.engineType} · ${m.category}" +
                if (m.notes.isNotBlank()) " · ${m.notes}" else ""

            val enrolled = m.scorerJson != null
            val total = registry.totalLabeledClips(m.id)
            row.findViewById<TextView>(R.id.rowStatus).text = buildString {
                append(if (enrolled) "Enrolled" else "Not enrolled")
                append("  ·  ${m.checkCount} checks")
                append("  ·  $total training clips")
                if (m.lastVerdict != null) append("  ·  last: ${m.lastVerdict.lowercase()}")
            }

            val pending = clipStore.pendingCount(m.id)
            val pendingView = row.findViewById<TextView>(R.id.rowPending)
            if (pending > 0) {
                pendingView.visibility = View.VISIBLE
                pendingView.text = "$pending recording${if (pending == 1) "" else "s"} waiting for a verdict — tap to label"
            }

            row.setOnClickListener {
                if (pending > 0) showPendingQueue(m.id) else selectAndReturn(m.id)
            }
            row.setOnLongClickListener { confirmDelete(m.id); true }
            listContainer.addView(row)
        }
    }

    private fun selectAndReturn(id: String) {
        setResult(Activity.RESULT_OK, Intent().putExtra(EXTRA_SELECTED_ID, id))
        finish()
    }

    private fun confirmDelete(id: String) {
        AlertDialog.Builder(this)
            .setTitle("Remove $id?")
            .setMessage("Its enrolled fingerprint is deleted. Training clips already saved to disk are kept — they still belong to the corpus.")
            .setPositiveButton("Remove") { _, _ -> registry.delete(id); refreshList() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ------------------------------------------------------------ add machine

    private fun showAddMachineDialog() {
        val view = LayoutInflater.from(this).inflate(R.layout.dialog_add_machine, null)
        val nameInput = view.findViewById<EditText>(R.id.nameInput)
        val engineGroup = view.findViewById<RadioGroup>(R.id.engineTypeGroup)
        val categorySpinner = view.findViewById<Spinner>(R.id.categorySpinner)
        val notesInput = view.findViewById<EditText>(R.id.notesInput)
        categorySpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, MachineRegistry.Category.ALL)

        AlertDialog.Builder(this)
            .setTitle("Add a machine")
            .setView(view)
            .setPositiveButton("Add") { _, _ ->
                val id = nameInput.text.toString().trim()
                if (id.isEmpty()) return@setPositiveButton
                val engineType = when (engineGroup.checkedRadioButtonId) {
                    R.id.radioPetrol -> MachineRegistry.EngineType.PETROL
                    R.id.radioDiesel -> MachineRegistry.EngineType.DIESEL
                    else -> MachineRegistry.EngineType.UNKNOWN
                }
                val category = categorySpinner.selectedItem as? String ?: MachineRegistry.Category.OTHER
                val added = registry.addMachine(id, engineType, category, notesInput.text.toString().trim(), MachineRegistry.nowIso())
                if (added) selectAndReturn(id) else {
                    android.widget.Toast.makeText(this, "\"$id\" already exists", android.widget.Toast.LENGTH_SHORT).show()
                    refreshList()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // --------------------------------------------------------- pending queue

    private fun showPendingQueue(machineId: String) {
        val pending = clipStore.listPending(machineId)
        if (pending.isEmpty()) { selectAndReturn(machineId); return }

        val names = pending.map { "Recorded ${it.recordedAtUtc.take(16).replace('T', ' ')}  —  model said: ${it.provisionalTier.lowercase()}" }
        AlertDialog.Builder(this)
            .setTitle("$machineId — recordings awaiting a verdict")
            .setItems(names.toTypedArray()) { _, which -> showLabelDialog(pending[which]) }
            .setNegativeButton("Not now") { _, _ -> selectAndReturn(machineId) }
            .show()
    }

    private fun showLabelDialog(pending: LabeledClipStore.PendingClip) {
        val view = LayoutInflater.from(this).inflate(R.layout.dialog_label_clip, null)
        val group = view.findViewById<RadioGroup>(R.id.labelKindGroup)
        val radioFaulty = view.findViewById<android.widget.RadioButton>(R.id.radioFaulty)
        val spinner = view.findViewById<Spinner>(R.id.faultSpinner)
        val verdictInput = view.findViewById<EditText>(R.id.verdictInput)
        spinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, FaultLabel.faultyDisplayOptions())
        group.setOnCheckedChangeListener { _, _ ->
            val faulty = radioFaulty.isChecked
            spinner.visibility = if (faulty) View.VISIBLE else View.GONE
            verdictInput.visibility = if (faulty) View.VISIBLE else View.GONE
        }

        AlertDialog.Builder(this)
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
                refreshList()
            }
            .setNeutralButton("Discard recording") { _, _ ->
                clipStore.discardPending(pending)
                refreshList()
            }
            .setNegativeButton("Later", null)
            .show()
    }
}
