package com.injini.app

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/**
 * What Enrol, Check and Add verdict actually do, in that order — not a pitch
 * for the idea of acoustic monitoring. Shown once automatically after the
 * first machine is added ([MachineForms]), and any time after from
 * "How this works" on the home screen.
 */
class OnboardingActivity : AppCompatActivity() {

    companion object {
        private const val PREFS = "injini"
        private const val KEY_SEEN = "onboarding_seen"
        private const val APK_URL = "https://github.com/rapha18th/injini/releases/latest/download/app-debug.apk"

        fun markSeen(context: Context) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_SEEN, true).apply()
        }

        fun hasBeenSeen(context: Context): Boolean =
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_SEEN, false)

        fun start(context: Context) {
            context.startActivity(Intent(context, OnboardingActivity::class.java))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_onboarding)
        applyInsets()
        findViewById<View>(R.id.onboardingDoneButton).setOnClickListener {
            markSeen(this)
            finish()
        }
        findViewById<View>(R.id.onboardingShareLink).setOnClickListener {
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, "Injini, offline acoustic engine monitoring: $APK_URL")
            }
            startActivity(Intent.createChooser(send, "Share Injini"))
        }
    }

    private fun applyInsets() {
        val root = findViewById<View>(R.id.onboardingRoot)
        val l = root.paddingLeft; val t = root.paddingTop; val r = root.paddingRight; val b = root.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(l + bars.left, t + bars.top, r + bars.right, b + bars.bottom)
            insets
        }
    }
}
