package fr.yawendaba.flutter_whisper_ggml

import android.util.Log
import com.whispercpp.whisper.WhisperContext
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WhisperPlugin : FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel
    // Cache the Whisper context to avoid reloading the model on each call
    private var cachedContext: WhisperContext? = null
    private var cachedModelPath: String? = null
    private val contextMutex = Mutex() // Mutex for thread-safe context access

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_whisper_ggml")
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        // Release cached context when plugin is detached
        CoroutineScope(Dispatchers.IO).launch {
            contextMutex.withLock {
                cachedContext?.release()
                cachedContext = null
                cachedModelPath = null
            }
        }
    }

    private suspend fun getOrCreateContext(modelPath: String): WhisperContext {
        return withContext(Dispatchers.IO) {
            contextMutex.withLock {
                // Return cached context if same model path
                if (cachedContext != null && cachedModelPath == modelPath) {
                    Log.d("WhisperPlugin", "Reusing cached Whisper context")
                    return@withLock cachedContext!!
                }
                
                // Release old context if model path changed
                if (cachedContext != null) {
                    Log.d("WhisperPlugin", "Model path changed, releasing old context")
                    cachedContext?.release()
                    cachedContext = null
                }
                
                // Load new model
                Log.d("WhisperPlugin", "Loading Whisper model from: $modelPath")
                val context = WhisperContext.createContextFromFile(modelPath)
                cachedContext = context
                cachedModelPath = modelPath
                Log.d("WhisperPlugin", "Model loaded and cached")
                return@withLock context
            }
        }
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
                        
                        // Load model only once and cache it
                        val whisper = getOrCreateContext(modelPath)
                        val text = whisper.transcribeData(floatArray, printTimestamp = false)
                        
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

        // Don't trim silence - let Whisper handle it with its default parameters
        // Trim silence can remove important audio content
        Log.d("WhisperPlugin", "Audio samples: ${floatArray.size}")

        return floatArray
    }
}

