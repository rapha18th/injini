package com.injini.app

import android.os.Build
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import kotlin.concurrent.thread

/**
 * One button: "Sync now". The relay URL and upload key are bundled in the
 * app (see [HfSync]'s own doc comment for why that's safe here) — nobody
 * distributing or using this screen should need to know Hugging Face, a
 * relay, or a key exist at all. See `hf_space/README.md` for what actually
 * happens on the other end of this tap.
 */
class HfSyncActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_hf_sync)
        applyInsets()
        findViewById<View>(R.id.hfUploadButton).setOnClickListener { syncNow() }
    }

    private fun applyInsets() {
        val root = findViewById<View>(R.id.hfSyncRoot)
        val l = root.paddingLeft; val t = root.paddingTop; val r = root.paddingRight; val b = root.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(l + bars.left, t + bars.top, r + bars.right, b + bars.bottom)
            insets
        }
    }

    private fun syncNow() {
        val progress = AlertDialog.Builder(this)
            .setTitle("Syncing")
            .setMessage("Sending this phone's recordings to Injini Cloud…")
            .setCancelable(false)
            .create()
        progress.show()

        val deviceLabel = "${Build.MANUFACTURER}_${Build.MODEL}".replace(Regex("[^A-Za-z0-9_-]"), "_")

        thread {
            var zip: File? = null
            try {
                zip = DatasetExporter.zipToCache(this)
                HfSync.upload(zip, HfSync.savedRelayUrl(this), HfSync.savedApiKey(this), deviceLabel)
                runOnUiThread {
                    progress.dismiss()
                    AlertDialog.Builder(this)
                        .setTitle("Synced")
                        .setMessage("Sent to Injini Cloud. Thanks for the data.")
                        .setPositiveButton("OK", null)
                        .show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    progress.dismiss()
                    AlertDialog.Builder(this)
                        .setTitle("Couldn't sync")
                        .setMessage("Check the connection and try again.\n\n${e.message ?: ""}")
                        .setPositiveButton("OK", null)
                        .show()
                }
            } finally {
                zip?.delete()
            }
        }
    }
}
