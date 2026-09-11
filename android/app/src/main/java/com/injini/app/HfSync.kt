package com.injini.app

import android.content.Context
import java.io.File
import java.util.concurrent.TimeUnit
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody

/**
 * Uploads to the phone's own "Injini Relay" (see `hf_space/` in the repo) —
 * never straight to Hugging Face. A bundled Hugging Face write token would
 * be extractable from any decompiled APK and would hand out real write
 * access to the dataset repo; this app never holds one. Instead it holds a
 * separate, low-privilege upload key the relay checks before it does
 * anything — if that key leaks, the worst it buys someone is the ability to
 * open a pull request against the dataset, never to write to it directly,
 * since the relay always stages uploads as a PR for a human to review.
 *
 * Purely user-triggered — this app is offline-only except for this one
 * explicit action.
 */
object HfSync {
    private const val PREFS = "injini_hf"
    private const val KEY_URL = "relay_url"
    private const val KEY_API_KEY = "api_key"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    fun savedRelayUrl(context: Context): String = prefs(context).getString(KEY_URL, "") ?: ""
    fun savedApiKey(context: Context): String = prefs(context).getString(KEY_API_KEY, "") ?: ""

    fun save(context: Context, relayUrl: String, apiKey: String) {
        prefs(context).edit()
            .putString(KEY_URL, relayUrl.trim().trimEnd('/'))
            .putString(KEY_API_KEY, apiKey.trim())
            .apply()
    }

    private fun prefs(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** Runs on the calling thread — call from a background thread. Throws with a message meant to be shown to the user directly. */
    fun upload(zipFile: File, relayUrl: String, apiKey: String, deviceLabel: String): String {
        if (relayUrl.isBlank()) error("No relay URL saved yet.")
        if (apiKey.isBlank()) error("No upload key saved yet.")

        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("source", deviceLabel)
            .addFormDataPart("file", zipFile.name, zipFile.asRequestBody("application/zip".toMediaType()))
            .build()

        val request = Request.Builder()
            .url("${relayUrl.trim().trimEnd('/')}/api/upload")
            .addHeader("X-Api-Key", apiKey)
            .post(body)
            .build()

        client.newCall(request).execute().use { response ->
            val text = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                error("Upload failed (${response.code}): ${text.take(300).ifBlank { "no details from the server" }}")
            }
            return text
        }
    }
}
