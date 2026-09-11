package com.injini.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * On-device register of every machine this phone has enrolled or collected
 * training clips for, one JSON file, same shape as TapSense's VesselRegistry.
 *
 * A machine used to be a single hardcoded name with one scorer in
 * SharedPreferences. That could not support a fleet: a kombi rank or a
 * generator dealer needs many named machines, each with its own healthy
 * fingerprint, on one phone. The registry also tracks how many labelled
 * training clips have been collected per machine per label, because that
 * count is what "the training corpus is thin" turns into "here is a fleet
 * conversation's worth of real data" — see [LabeledClipStore].
 */
class MachineRegistry(context: Context) {

    data class Machine(
        val id: String,
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

    fun knownIds(): List<String> = all().map { it.id }.sortedBy { it.lowercase() }

    /** Creates the machine if it does not exist yet; otherwise leaves it untouched. */
    fun ensureExists(id: String, nowUtc: String) {
        val trimmed = id.trim()
        if (trimmed.isEmpty() || find(trimmed) != null) return
        val m = Machine(trimmed, nowUtc, null, 0, null, null, emptyMap())
        write(all() + m)
    }

    fun saveScorer(id: String, scorerJson: String) {
        val trimmed = id.trim()
        if (trimmed.isEmpty()) return
        val existing = find(trimmed)
        val merged = (existing ?: Machine(trimmed, nowIso(), null, 0, null, null, emptyMap()))
            .copy(scorerJson = scorerJson)
        replace(trimmed, merged)
    }

    fun recordCheck(id: String, verdict: String, nowUtc: String) {
        val trimmed = id.trim()
        val existing = find(trimmed) ?: return
        replace(trimmed, existing.copy(
            checkCount = existing.checkCount + 1,
            lastVerdict = verdict,
            lastCheckedAtUtc = nowUtc,
        ))
    }

    fun recordLabeledClip(id: String, corpusLabel: String, nowUtc: String) {
        val trimmed = id.trim()
        if (trimmed.isEmpty()) return
        val existing = find(trimmed) ?: Machine(trimmed, nowUtc, null, 0, null, null, emptyMap())
        val counts = existing.labeledCounts.toMutableMap()
        counts[corpusLabel] = (counts[corpusLabel] ?: 0) + 1
        replace(trimmed, existing.copy(labeledCounts = counts))
    }

    fun totalLabeledClips(id: String): Int = find(id)?.labeledCounts?.values?.sum() ?: 0

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
