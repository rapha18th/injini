package com.injini.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * A live level meter, drawn as scrolling bars in the same shape as the
 * app icon's five-bar waveform. Exists so a ten-second recording shows
 * something alive on screen instead of a status line that does not change
 * until the clip is over — the "is this thing stuck" problem.
 *
 * [pushLevel] is called from the capture thread once per audio chunk (roughly
 * every 15-30 ms); the view keeps the last [barCount] levels and redraws.
 * All drawing happens on the UI thread via [android.view.View.postInvalidate],
 * so [pushLevel] is safe to call from a background thread.
 */
class WaveformView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val barCount = 40
    private val levels = FloatArray(barCount)
    private var writeIndex = 0

    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFE0A64B.toInt() // injini_amber
    }
    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x33E0A64B // faint amber, the resting/empty bar
    }

    /** Clears the trace back to flat, called at the start of each recording. */
    fun reset() {
        levels.fill(0f)
        writeIndex = 0
        postInvalidate()
    }

    /** [level] is 0..1, roughly the chunk's peak amplitude. Thread-safe. */
    fun pushLevel(level: Float) {
        synchronized(levels) {
            levels[writeIndex % barCount] = level.coerceIn(0f, 1f)
            writeIndex++
        }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        val gap = w / barCount * 0.28f
        val barWidth = w / barCount - gap
        val minBarHeight = h * 0.06f

        val snapshot: FloatArray
        val startIdx: Int
        synchronized(levels) {
            snapshot = levels.copyOf()
            startIdx = writeIndex
        }

        for (i in 0 until barCount) {
            // Oldest bar on the left, most recent on the right, scrolling.
            val level = snapshot[(startIdx + i) % barCount]
            val barHeight = max(minBarHeight, level * h)
            val left = i * (barWidth + gap)
            val top = (h - barHeight) / 2f
            val paint = if (level > 0.02f) barPaint else trackPaint
            canvas.drawRoundRect(
                left, top, left + barWidth, top + barHeight,
                barWidth / 2f, barWidth / 2f, paint,
            )
        }
    }
}
