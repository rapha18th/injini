package com.injini.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * On-device register of every machine this phone has enrolled, one JSON
 * file, same shape as TapSense's VesselRegistry.
 *
 * Engine type and category are not cosmetic fields. A rod knock sounds
 * different on a diesel compression-ignition engine than on a petrol
 * spark-ignition one, and a generator's fixed governed speed makes its
 * order structure nothing like a vehicle's. A future retrain needs to know
 * which population a clip came from, or it will happily learn "diesel"
 * where it meant "faulty." These fields are copied onto every clip this
 * machine produces — see [LabeledClipStore].
 */
class MachineRegistry(context: Context) {

    object EngineType {
        const val PETROL = "Petrol"
        const val DIESEL = "Diesel"
        const val UNKNOWN = "Unknown"
        val ALL = listOf(PETROL, DIESEL, UNKNOWN)
    }

    object Category {
        const val VEHICLE = "Vehicle"
        const val GENERATOR = "Generator"
        const val PUMP = "Pump"
        const val OTHER = "Other"
        val ALL = listOf(VEHICLE, GENERATOR, PUMP, OTHER)
    }

    data class Machine(
        val id: String,
        val engineType: String,
        val category: String,
        val notes: String,
        val createdAtUtc: String,
        /** [AnomalyScorer.toJson], or null if this machine has not been enrolled yet. */
        val scorerJson: String?,
        val checkCount: Int,
        val lastVerdict: String?,
        val lastCheckedAtUtc: String?,
        /** corpus label -> number of training clips collected for this machine under it. */
        val labeledCounts: Map<String, Int>,
    )

    private val file = File(context.getExternalFilesDir(null), "InjiniData/machines.json")

    fun all(): List<Machine> {
        if (!file.exists()) return emptyList()
        return try {
            val root = JSONArray(file.readText())
            (0 until root.length()).map { i -> fromJson(root.getJSONObject(i)) }
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun find(id: String): Machine? = all().firstOrNull { it.id.equals(id.trim(), ignoreCase = true) }

    fun idExists(id: String): Boolean = find(id) != null

    /** Creates a new machine with full attributes. Overwrites nothing if the id already exists. */
    fun addMachine(id: String, engineType: String, category: String, notes: String, nowUtc: String): Boolean {
        val trimmed = id.trim()
        if (trimmed.isEmpty() || idExists(trimmed)) return false
        write(all() + Machine(trimmed, engineType, category, notes, nowUtc, null, 0, null, null, emptyMap()))
        return true
    }

    fun updateAttributes(id: String, engineType: String, category: String, notes: String) {
        val existing = find(id) ?: return
        replace(id, existing.copy(engineType = engineType, category = category, notes = notes))
    }

    fun saveScorer(id: String, scorerJson: String) {
        val existing = find(id) ?: return
        replace(id, existing.copy(scorerJson = scorerJson))
    }

    fun recordCheck(id: String, verdict: String, nowUtc: String) {
        val existing = find(id) ?: return
        replace(id, existing.copy(
            checkCount = existing.checkCount + 1,
            lastVerdict = verdict,
            lastCheckedAtUtc = nowUtc,
        ))
    }

    fun recordLabeledClip(id: String, corpusLabel: String, count: Int = 1) {
        val existing = find(id) ?: return
        val counts = existing.labeledCounts.toMutableMap()
        counts[corpusLabel] = (counts[corpusLabel] ?: 0) + count
        replace(id, existing.copy(labeledCounts = counts))
    }

    fun totalLabeledClips(id: String): Int = find(id)?.labeledCounts?.values?.sum() ?: 0

    fun delete(id: String) {
        write(all().filterNot { it.id.equals(id, ignoreCase = true) })
    }

    private fun replace(id: String, updated: Machine) {
        val others = all().filterNot { it.id.equals(id, ignoreCase = true) }
        write(others + updated)
    }

    private fun write(machines: List<Machine>) {
        file.parentFile?.mkdirs()
        val arr = JSONArray()
        machines.sortedBy { it.id.lowercase() }.forEach { arr.put(toJson(it)) }
        file.writeText(arr.toString(2))
    }

    private fun toJson(m: Machine) = JSONObject().apply {
        put("id", m.id)
        put("engine_type", m.engineType)
        put("category", m.category)
        put("notes", m.notes)
        put("created_at_utc", m.createdAtUtc)
        put("scorer_json", m.scorerJson ?: JSONObject.NULL)
        put("check_count", m.checkCount)
        put("last_verdict", m.lastVerdict ?: JSONObject.NULL)
        put("last_checked_at_utc", m.lastCheckedAtUtc ?: JSONObject.NULL)
        put("labeled_counts", JSONObject().apply { m.labeledCounts.forEach { (k, v) -> put(k, v) } })
    }

    private fun fromJson(o: JSONObject): Machine {
        val countsObj = o.optJSONObject("labeled_counts")
        val counts = mutableMapOf<String, Int>()
        countsObj?.keys()?.forEach { k -> counts[k] = countsObj.optInt(k, 0) }
        return Machine(
            id = o.optString("id"),
            engineType = o.optString("engine_type", EngineType.UNKNOWN),
            category = o.optString("category", Category.OTHER),
            notes = o.optString("notes", ""),
            createdAtUtc = o.optString("created_at_utc", ""),
            scorerJson = if (o.isNull("scorer_json")) null else o.optString("scorer_json"),
            checkCount = o.optInt("check_count", 0),
            lastVerdict = if (o.isNull("last_verdict")) null else o.optString("last_verdict"),
            lastCheckedAtUtc = if (o.isNull("last_checked_at_utc")) null else o.optString("last_checked_at_utc"),
            labeledCounts = counts,
        )
    }

    companion object {
        fun nowIso(): String = java.time.Instant.now().toString()
    }
}
