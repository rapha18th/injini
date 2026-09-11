package com.injini.app

import android.app.Activity
import android.view.LayoutInflater
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.Toast
import androidx.appcompat.app.AlertDialog

/**
 * The "add a machine" dialog, shared between [MainActivity] (so adding a
 * machine never requires leaving the home screen) and [MachineListActivity]
 * (the full fleet screen) — one form, one behaviour, instead of two copies
 * that could quietly drift apart.
 */
object MachineForms {

    fun showAddMachineDialog(activity: Activity, registry: MachineRegistry, onAdded: (String) -> Unit) {
        val view = LayoutInflater.from(activity).inflate(R.layout.dialog_add_machine, null)
        val nameInput = view.findViewById<EditText>(R.id.nameInput)
        val engineGroup = view.findViewById<RadioGroup>(R.id.engineTypeGroup)
        val categorySpinner = view.findViewById<Spinner>(R.id.categorySpinner)
        val notesInput = view.findViewById<EditText>(R.id.notesInput)
        categorySpinner.adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_dropdown_item, MachineRegistry.Category.ALL)

        AlertDialog.Builder(activity)
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
                if (added) onAdded(id) else Toast.makeText(activity, "\"$id\" already exists", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
