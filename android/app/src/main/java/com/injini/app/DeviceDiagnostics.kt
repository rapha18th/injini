package com.injini.app

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager

/**
 * Real, measured device-level numbers (RAM footprint, thermal status), not
 * asserted ones. Added specifically because "does INT8's smaller footprint
 * avoid the Low Memory Killer" and "does the SoC stay out of thermal
 * throttling" are answerable in part, and an unmeasured claim either way
 * isn't good enough for a submission.
 */
object DeviceDiagnostics {

    /** Process PSS (proportional set size) in KB — real committed memory, not virtual size. */
    fun pssKb(): Int {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss
    }

    /**
     * Coarse OS-reported thermal status. Requires API 29+ (minSdk here is
     * 26) — reports that plainly rather than silently returning nothing on
     * older devices.
     */
    fun thermalStatus(context: Context): String {
        if (Build.VERSION.SDK_INT < 29) {
            return "unavailable (needs Android 10 / API 29+, device is API ${Build.VERSION.SDK_INT})"
        }
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
            ?: return "unavailable (no PowerManager)"
        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "NONE"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
            else -> "UNKNOWN"
        }
    }

    /**
     * Instantaneous battery current draw in microamps, read via Android's
     * BatteryManager API from inside the app process. Unlike
     * /sys/class/power_supply/battery/current_now (confirmed Permission
     * denied for the shell user on this device), app-level access goes
     * through the framework, not raw sysfs, so it isn't blocked the same
     * way. Returns null if the device doesn't support this property
     * (Int.MIN_VALUE) rather than a fabricated number — callers must report
     * that explicitly instead of silently falling back.
     */
    fun currentDrawMicroamps(context: Context): Int? {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager ?: return null
        val value = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        return if (value == Int.MIN_VALUE) null else value
    }
}
