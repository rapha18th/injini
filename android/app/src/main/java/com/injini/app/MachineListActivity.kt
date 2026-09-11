package com.injini.app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/**
 * The fleet screen. A kombi rank or a generator dealer has many machines on
 * one phone, each with its own attributes and its own contribution to the
 * training corpus — this is where that is managed.
 *
 * Tapping a machine's name/status area always selects it and returns to
 * [MainActivity] ([Activity.RESULT_OK] with [EXTRA_SELECTED_ID]) — that used
 * to be hijacked into opening the pending-verdict queue whenever one
 * existed, which made picking an already-enrolled machine impossible without
 * first dismissing something worded like a decline. Reviewing pending
 * recordings is now its own clearly labelled button on the row, wired
 * through the same [VerdictFlow] the home screen's "Add verdict" button
 * uses. Long-pressing a row offers to delete that machine's registry entry;
 * its recordings on disk are untouched.
 */
class MachineListActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_SELECTED_ID = "selected_machine_id"
    }

    private lateinit var registry: MachineRegistry
    private lateinit var clipStore: LabeledClipStore
    private lateinit var listContainer: LinearLayout
    private lateinit var verdictFlow: VerdictFlow

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_machine_list)
        applyInsets()
        registry = MachineRegistry(this)
        clipStore = LabeledClipStore(this)
        verdictFlow = VerdictFlow(this, registry, clipStore, onChanged = { refreshList() })
        listContainer = findViewById(R.id.machineList)
        findViewById<View>(R.id.addMachineButton).setOnClickListener {
            MachineForms.showAddMachineDialog(this, registry) { id -> selectAndReturn(id) }
        }
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
            row.findViewById<View>(R.id.rowSelectArea).setOnClickListener { selectAndReturn(m.id) }

            val pending = clipStore.pendingCount(m.id)
            val pendingView = row.findViewById<TextView>(R.id.rowPending)
            if (pending > 0) {
                pendingView.visibility = View.VISIBLE
                pendingView.text = "⚠  $pending recording${if (pending == 1) "" else "s"} need${if (pending == 1) "s" else ""} a verdict  —  Add verdict ›"
                pendingView.setOnClickListener { verdictFlow.showQueue(m.id) }
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
}
