package fr.yawendaba.flutter_whisper_ggml

import android.util.Log
import com.whispercpp.whisper.WhisperContext
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WhisperPlugin : FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_whisper_ggml")
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "transcribe" -> {
                Log.d("WhisperPlugin", "transcribe called")
                val filePath = call.argument<String>("filePath")
                val modelPath = call.argument<String>("modelPath")

                if (filePath == null || modelPath == null) {
                    result.error("INVALID_ARGUMENT", "filePath and modelPath are required", null)
                    return
                }

                CoroutineScope(Dispatchers.IO).launch {
                    try {
                        val floatArray = readWavFile(filePath)
                        val whisper = WhisperContext.createContextFromFile(modelPath)
                        val text = whisper.transcribeData(floatArray, printTimestamp = false)
                        whisper.release()
                        
                        withContext(Dispatchers.Main) {
                            result.success(text)
                        }
                    } catch (e: Exception) {
                        Log.e("WhisperPlugin", "Error during transcription", e)
                        withContext(Dispatchers.Main) {
                            result.error("TRANSCRIPTION_ERROR", e.message, null)
                        }
                    }
                }
            }
            else -> {
                result.notImplemented()
            }
        }
    }

    private fun readWavFile(filePath: String): FloatArray {
        val file = File(filePath)
        val bytes = file.readBytes()

        // Parse WAV header
        if (bytes.size < 44) {
            throw IllegalArgumentException("WAV file too small")
        }

        // Check RIFF header
        val riff = String(bytes.sliceArray(0..3))
        if (riff != "RIFF") {
            throw IllegalArgumentException("Invalid WAV file: missing RIFF header")
        }

        // Check WAVE header
        val wave = String(bytes.sliceArray(8..11))
        if (wave != "WAVE") {
            throw IllegalArgumentException("Invalid WAV file: missing WAVE header")
        }

        // Parse chunks
        var pos = 12
        var sampleRate: Int? = null
        var numChannels: Int? = null
        var bitsPerSample: Int? = null
        var dataStart: Int? = null
        var dataSize: Int? = null

        while (pos < bytes.size - 8) {
            val chunkId = String(bytes.sliceArray(pos until pos + 4))
            val chunkSize = ByteBuffer.wrap(bytes, pos + 4, 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .int

            if (chunkId == "fmt ") {
                // Parse fmt chunk
                val audioFormat = ByteBuffer.wrap(bytes, pos + 8, 2)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .short.toInt()
                numChannels = ByteBuffer.wrap(bytes, pos + 10, 2)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .short.toInt()
                sampleRate = ByteBuffer.wrap(bytes, pos + 12, 4)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .int
                bitsPerSample = ByteBuffer.wrap(bytes, pos + 22, 2)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .short.toInt()

                if (audioFormat != 1) {
                    throw IllegalArgumentException("Unsupported audio format: $audioFormat (only PCM supported)")
                }
            } else if (chunkId == "data") {
                dataStart = pos + 8
                dataSize = chunkSize
                break
            }

            pos += 8 + chunkSize
            if (chunkSize % 2 == 1) pos++
        }

        if (sampleRate == null || numChannels == null || bitsPerSample == null) {
            throw IllegalArgumentException("Missing fmt chunk in WAV file")
        }

        if (dataStart == null || dataSize == null) {
            throw IllegalArgumentException("Missing data chunk in WAV file")
        }

        // Validate format
        if (numChannels != 1 || sampleRate != 16000 || bitsPerSample != 16) {
            throw IllegalArgumentException("WAV must be PCM16 mono @16kHz")
        }

        // Read PCM16 samples
        val bytesPerSample = bitsPerSample / 8
        val totalSamples = dataSize / bytesPerSample / numChannels

        // Convert to float32
        val floatArray = FloatArray(totalSamples)
        for (i in 0 until totalSamples) {
            val sampleOffset = dataStart + i * bytesPerSample * numChannels
            val sampleInt = ByteBuffer.wrap(bytes, sampleOffset, 2)
                .order(ByteOrder.LITTLE_ENDIAN)
                .short.toInt()
            floatArray[i] = sampleInt / 32768.0f
        }

        // Trim silence
        val trimmed = trimSilence(floatArray)
        Log.d("WhisperPlugin", "Audio samples: ${floatArray.size} -> ${trimmed.size} (after trim)")

        return trimmed
    }

    private fun trimSilence(samples: FloatArray, threshold: Float = 0.02f): FloatArray {
        var i0 = 0
        var i1 = samples.size

        while (i0 < i1 && kotlin.math.abs(samples[i0]) < threshold) i0++
        while (i1 > i0 && kotlin.math.abs(samples[i1 - 1]) < threshold) i1--

        return if (i0 > 0 || i1 < samples.size) {
            samples.sliceArray(i0 until i1)
        } else {
            samples
        }
    }
}

