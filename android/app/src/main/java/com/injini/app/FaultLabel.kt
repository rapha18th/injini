package com.injini.app

/**
 * The label set for field-collected training clips, aligned with the
 * fault-ID head's actual training corpus (see prepare_engine_sounds.py and
 * the class list in models/faultid_mn10_as_fp32.json) so a future retrain can
 * merge field data with the public corpus under the same class names rather
 * than creating a parallel, incompatible taxonomy.
 *
 * Each display label maps to a "corpus key", the exact string the training
 * pipeline already uses. One of those keys, "Engine kanocking", carries a
 * typo from the public dataset itself — kept here on purpose, not fixed,
 * because a clean respelling would silently create a 13th near-duplicate
 * class instead of joining the existing one. The display label stays
 * correctly spelled; only the stored corpus key preserves the original.
 *
 * "Other" is not a corpus key that exists yet. A mechanic's verdict that does
 * not match any known class is still saved, under a corpus key built from the
 * mechanic's own words, because a fault this taxonomy has no name for yet is
 * exactly the kind of finding a fixed picklist would otherwise throw away.
 */
object FaultLabel {

    const val HEALTHY_DISPLAY = "Healthy (after service)"
    const val HEALTHY_CORPUS_KEY = "Normal"

    const val OTHER_DISPLAY = "Other — describe it below"

    /** display label -> corpus key, for every fault class the current classifier already knows. */
    val KNOWN_FAULTS: LinkedHashMap<String, String> = linkedMapOf(
        "Rod knock" to "Rod knock",
        "Piston slap" to "piston slap",
        "Valve tapping" to "Valve tapping",
        "Engine knocking" to "Engine kanocking",
        "Exhaust leak" to "Exhaust leak",
        "Vacuum leak" to "Vacuum leak",
        "Timing belt noise" to "Timing belt",
        "Worn pulley noise" to "Worn Pulley Noise",
        "Chain noise" to "Chain Noise",
        "Alternator bearing noise" to "Alternator Bearing Noise",
        "Crankshaft bearing noise" to "crankshaft bearing noise",
    )

    /** Every display option offered in the faulty-machine picker, known faults then "Other" last. */
    fun faultyDisplayOptions(): List<String> = KNOWN_FAULTS.keys.toList() + OTHER_DISPLAY

    /**
     * Turns a UI selection into the corpus key that gets written to the
     * manifest. For "Other" the mechanic's own verdict text becomes the key,
     * trimmed and title-cased so near-identical verdicts ("worn cv joint",
     * "Worn CV joint") collapse into one class instead of scattering.
     */
    fun corpusKeyFor(displayLabel: String, mechanicVerdict: String): String {
        if (displayLabel == OTHER_DISPLAY) {
            val cleaned = mechanicVerdict.trim()
            return if (cleaned.isEmpty()) "Unspecified fault" else cleaned
        }
        return KNOWN_FAULTS[displayLabel] ?: displayLabel
    }
}
