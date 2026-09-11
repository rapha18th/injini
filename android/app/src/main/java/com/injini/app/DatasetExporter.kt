package com.injini.app

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import java.io.BufferedOutputStream
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Zips every recording this phone has collected — the confirmed corpus and
 * whatever's still in the pending queue, so nothing is left behind because
 * it hadn't been given a verdict yet — into one file a person can hand off
 * without a data connection: saved to Downloads so any file manager sees
 * it, and returned as a content:// Uri ready for a share sheet, a WhatsApp
 * attachment being the actual point rather than an afterthought.
 *
 * Android 10+ writes through MediaStore, the only way onto shared storage
 * once scoped storage applies — no FileProvider needed there, MediaStore's
 * own Uri is already shareable. Older devices (this app's minSdk is 26, and
 * a rugged budget phone still on Android 8/9 is a realistic device in this
 * app's actual market) fall back to a direct file write plus FileProvider.
 */
object DatasetExporter {

    fun export(context: Context): Uri {
        val name = zipName()
        val root = corpusRoot(context)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            exportViaMediaStore(context, name, root)
        } else {
            exportViaLegacyFile(context, name, root)
        }
    }

    /** Zips the corpus into the app's private cache instead of shared storage — for [HfSync], which uploads the bytes directly rather than saving/sharing a file. Caller deletes it once done. */
    fun zipToCache(context: Context): File {
        val file = File(context.cacheDir, zipName())
        file.outputStream().use { out -> zip(corpusRoot(context), out) }
        return file
    }

    private fun corpusRoot(context: Context) = File(context.getExternalFilesDir(null), "InjiniLabeled")

    private fun zipName() = "injini_dataset_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.zip"

    private fun exportViaMediaStore(context: Context, name: String, root: File): Uri {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, name)
            put(MediaStore.Downloads.MIME_TYPE, "application/zip")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: error("Could not create $name in Downloads")
        resolver.openOutputStream(uri)?.use { out -> zip(root, out) }
            ?: error("Could not open $name for writing")
        val done = ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }
        resolver.update(uri, done, null, null)
        return uri
    }

    private fun exportViaLegacyFile(context: Context, name: String, root: File): Uri {
        val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        downloads.mkdirs()
        val file = File(downloads, name)
        file.outputStream().use { out -> zip(root, out) }
        return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    }

    private fun zip(root: File, out: OutputStream) {
        ZipOutputStream(BufferedOutputStream(out)).use { zos ->
            if (root.exists()) zipDir(root, root, zos)
        }
    }

    private fun zipDir(base: File, dir: File, zos: ZipOutputStream) {
        dir.listFiles()?.sortedBy { it.name }?.forEach { f ->
            val rel = f.relativeTo(base).path.replace('\\', '/')
            if (f.isDirectory) {
                zipDir(base, f, zos)
            } else {
                zos.putNextEntry(ZipEntry(rel))
                f.inputStream().use { it.copyTo(zos) }
                zos.closeEntry()
            }
        }
    }
}
