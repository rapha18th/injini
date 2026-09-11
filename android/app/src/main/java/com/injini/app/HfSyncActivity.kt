package com.injini.app

import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import kotlin.concurrent.thread

/**
 * Settings + trigger for [HfSync]: paste the relay's URL and its upload key
 * once, then "Upload now" zips the corpus fresh each time and sends it. See
 * [HfSync] and `hf_space/README.md` for why this goes through a small relay
 * this project runs rather than straight to Hugging Face — the short
 * version is that a phone can't safely carry a real Hugging Face write
 * token, since anyone could pull it back out of the APK.
 */
class HfSyncActivity : AppCompatActivity() {

    private lateinit var urlInput: EditText
    private lateinit var keyInput: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_hf_sync)
        applyInsets()

        urlInput = findViewById(R.id.hfUrlInput)
        keyInput = findViewById(R.id.hfKeyInput)
        urlInput.setText(HfSync.savedRelayUrl(this))
        keyInput.setText(HfSync.savedApiKey(this))

        findViewById<View>(R.id.hfSaveButton).setOnClickListener {
            HfSync.save(this, urlInput.text.toString(), keyInput.text.toString())
            Toast.makeText(this, "Saved", Toast.LENGTH_SHORT).show()
        }
        findViewById<View>(R.id.hfUploadButton).setOnClickListener { uploadNow() }
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

    private fun uploadNow() {
        val relayUrl = urlInput.text.toString().trim()
        val apiKey = keyInput.text.toString().trim()
        if (relayUrl.isBlank() || apiKey.isBlank()) {
            Toast.makeText(this, "Enter both the relay URL and the upload key first.", Toast.LENGTH_LONG).show()
            return
        }
        HfSync.save(this, relayUrl, apiKey)

        val progress = AlertDialog.Builder(this)
            .setTitle("Uploading")
            .setMessage("Zipping the corpus and sending it to the relay…")
            .setCancelable(false)
            .create()
        progress.show()

        val deviceLabel = "${Build.MANUFACTURER}_${Build.MODEL}".replace(Regex("[^A-Za-z0-9_-]"), "_")

        thread {
            var zip: File? = null
            try {
                zip = DatasetExporter.zipToCache(this)
                val response = HfSync.upload(zip, relayUrl, apiKey, deviceLabel)
                runOnUiThread {
                    progress.dismiss()
                    AlertDialog.Builder(this)
                        .setTitle("Uploaded")
                        .setMessage(response)
                        .setPositiveButton("OK", null)
                        .show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    progress.dismiss()
                    AlertDialog.Builder(this)
                        .setTitle("Upload failed")
                        .setMessage(e.message ?: "Unknown error")
                        .setPositiveButton("OK", null)
                        .show()
                }
            } finally {
                zip?.delete()
            }
        }
    }
}
