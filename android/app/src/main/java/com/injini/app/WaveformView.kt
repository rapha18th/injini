package com.injini.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.sin

/**
 * A live level meter, drawn as scrolling bars in the same shape as the app
 * icon's five-bar waveform, plus a pulsing REC dot. Exists so a ten-second
 * recording shows something alive on screen instead of a status line that
 * does not change until the clip is over.
 *
 * The first version only reacted to real microphone level, and in a quiet
 * room that meant the bars sat almost flat — technically live, but
 * indistinguishable from stuck at a glance. [start] now runs a continuous
 * idle ripple (a slow travelling sine wave across the bars) and a breathing
 * REC dot independent of the microphone signal; real audio rides on top of
 * that floor rather than being the only source of motion. There is never a
 * moment while recording where nothing on screen is moving.
 */
class WaveformView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val barCount = 40
    private val levels = FloatArray(barCount)
    private var writeIndex = 0

    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFE0A64B.toInt() } // injini_amber
    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x33E0A64B }
    private val recPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFD4694E.toInt() } // injini_red

    private val handler = Handler(Looper.getMainLooper())
    private var phase = 0f
    private var running = false
    private val idleTick = object : Runnable {
        override fun run() {
            if (!running) return
            phase += 0.35f
            postInvalidate()
            handler.postDelayed(this, 45L)
        }
    }

    /** Clears the trace and starts the idle ripple + REC pulse. Call when a recording begins. */
    fun start() {
        levels.fill(0f)
        writeIndex = 0
        phase = 0f
        if (!running) {
            running = true
            handler.post(idleTick)
        }
    }

    /** Stops the idle animation. Call when a recording ends and the view is hidden. */
    fun stop() {
        running = false
        handler.removeCallbacks(idleTick)
    }

    /** Backwards-compatible alias for [start]. */
    fun reset() = start()

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
            // A travelling ripple across the bars, always present while
            // recording, so silence never reads as "nothing is happening".
            val idle = 0.12f + 0.08f * sin(phase * 0.12f + i * 0.35f).toFloat()
            val effective = max(idle, level)
            val barHeight = max(minBarHeight, effective * h)
            val left = i * (barWidth + gap)
            val top = (h - barHeight) / 2f
            val paint = if (effective > 0.16f) barPaint else trackPaint
            canvas.drawRoundRect(
                left, top, left + barWidth, top + barHeight,
                barWidth / 2f, barWidth / 2f, paint,
            )
        }

        // A breathing REC dot in the top-right corner — the same visual
        // language as a voice recorder or a camera app, unmistakable at a
        // glance even before anyone reads the status text.
        val dotRadius = h * 0.05f * (0.7f + 0.3f * sin(phase * 0.2f).toFloat())
        canvas.drawCircle(w - dotRadius - 6f, dotRadius + 6f, dotRadius, recPaint)
    }
}
