import Flutter
import whisper
import Foundation
import AVFoundation

// Declare C bridge functions from WhisperBridge.mm
@_silgen_name("whisper_init_bridge")
func whisper_init_bridge(_ path: UnsafePointer<CChar>) -> OpaquePointer?

@_silgen_name("whisper_full_bridge")
func whisper_full_bridge(_ ctx: OpaquePointer?, _ samples: UnsafeMutablePointer<Float>?, _ n: Int32) -> Int32

@_silgen_name("whisper_free_bridge")
func whisper_free_bridge(_ ctx: OpaquePointer?)

@_silgen_name("whisper_get_result_bridge")
func whisper_get_result_bridge(_ ctx: OpaquePointer?) -> UnsafePointer<CChar>?

@_silgen_name("whisper_free_cstr_bridge")
func whisper_free_cstr_bridge(_ s: UnsafePointer<CChar>?)

@objc(WhisperPlugin)
public class WhisperPlugin: NSObject, FlutterPlugin {
    private var cachedContext: OpaquePointer? = nil
    private var cachedModelPath: String? = nil
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        // 👇 Empêche le linker de supprimer whisper.xcframework
        EnforceBinding.dummyMethodToEnforceBundling()
        
        let channel = FlutterMethodChannel(name: "flutter_whisper_ggml", binaryMessenger: registrar.messenger())
        let instance = WhisperPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "transcribe":
            guard let args = call.arguments as? [String: Any],
                  let filePath = args["filePath"] as? String,
                  let modelPath = args["modelPath"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "filePath and modelPath are required", details: nil))
                return
            }
            
            transcribeAsync(filePath: filePath, modelPath: modelPath, result: result)
        case "transcribeData":
            guard let args = call.arguments as? [String: Any],
                  let modelPath = args["modelPath"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "modelPath and audioData are required", details: nil))
                return
            }
            
            if let doubles = args["audioData"] as? [Double] {
                let floatSamples = doubles.map { Float($0) }
                transcribeBuffer(modelPath: modelPath, samples: floatSamples, result: result)
            } else if let typed = args["audioData"] as? FlutterStandardTypedData {
                let byteCount = typed.data.count
                if byteCount % MemoryLayout<Float>.size != 0 {
                    result(FlutterError(code: "INVALID_ARGUMENT", message: "audioData has invalid byte length", details: nil))
                    return
                }
                let count = byteCount / MemoryLayout<Float>.size
                var floatSamples = [Float](repeating: 0, count: count)
                _ = floatSamples.withUnsafeMutableBytes { mutablePointer in
                    typed.data.copyBytes(to: mutablePointer)
                }
                transcribeBuffer(modelPath: modelPath, samples: floatSamples, result: result)
            } else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Unsupported audioData format", details: nil))
            }
            
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func transcribeAsync(filePath: String, modelPath: String, result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "ERROR", message: "Plugin deallocated", details: nil))
                }
                return
            }
            
            do {
                // Load or reuse context
                let context = try self.getOrCreateContext(modelPath: modelPath)
                
                // Read WAV file
                let audioData = try self.readWavFile(filePath: filePath)
                
                // Transcribe
                let transcript = try self.transcribe(context: context, audioData: audioData)
                
                DispatchQueue.main.async {
                    result(transcript)
                }
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(code: "TRANSCRIPTION_ERROR", message: error.localizedDescription, details: nil))
                }
            }
        }
    }
    
    private func transcribeBuffer(modelPath: String, samples: [Float], result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "ERROR", message: "Plugin deallocated", details: nil))
                }
                return
            }
            
            do {
                let context = try self.getOrCreateContext(modelPath: modelPath)
                let transcript = try self.transcribe(context: context, audioData: samples)
                
                DispatchQueue.main.async {
                    result(transcript)
                }
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(code: "TRANSCRIPTION_ERROR", message: error.localizedDescription, details: nil))
                }
            }
        }
    }
    
    private func getOrCreateContext(modelPath: String) throws -> OpaquePointer {
        // Reuse cached context if same model path
        if let cached = cachedContext, cachedModelPath == modelPath {
            return cached
        }
        
        // Release old context if model path changed
        if let oldContext = cachedContext {
            whisper_free_bridge(oldContext)
            cachedContext = nil
            cachedModelPath = nil
        }
        
        // Load new model
        guard let context = whisper_init_bridge(modelPath) else {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to load Whisper model"])
        }
        
        cachedContext = context
        cachedModelPath = modelPath
        
        return context
    }
    
    private func readWavFile(filePath: String) throws -> [Float] {
        let url = URL(fileURLWithPath: filePath)
        let data = try Data(contentsOf: url)
        
        guard data.count >= 44 else {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "WAV file too small"])
        }
        
        // Check RIFF header
        let riff = String(data: data[0..<4], encoding: .ascii)
        guard riff == "RIFF" else {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid WAV file: missing RIFF header"])
        }
        
        // Check WAVE header
        let wave = String(data: data[8..<12], encoding: .ascii)
        guard wave == "WAVE" else {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid WAV file: missing WAVE header"])
        }
        
        // Parse chunks
        var pos = 12
        var sampleRate: Int? = nil
        var numChannels: Int? = nil
        var bitsPerSample: Int? = nil
        var dataStart: Int? = nil
        var dataSize: Int? = nil
        
        while pos < data.count - 8 {
            let chunkId = String(data: data[pos..<pos+4], encoding: .ascii) ?? ""
            let chunkSize = data.withUnsafeBytes { bytes in
                bytes.load(fromByteOffset: pos + 4, as: UInt32.self).littleEndian
            }
            
            if chunkId == "fmt " {
                // Parse fmt chunk
                let audioFormat = data.withUnsafeBytes { bytes in
                    Int(bytes.load(fromByteOffset: pos + 8, as: UInt16.self).littleEndian)
                }
                numChannels = data.withUnsafeBytes { bytes in
                    Int(bytes.load(fromByteOffset: pos + 10, as: UInt16.self).littleEndian)
                }
                sampleRate = data.withUnsafeBytes { bytes in
                    Int(bytes.load(fromByteOffset: pos + 12, as: UInt32.self).littleEndian)
                }
                bitsPerSample = data.withUnsafeBytes { bytes in
                    Int(bytes.load(fromByteOffset: pos + 22, as: UInt16.self).littleEndian)
                }
                
                if audioFormat != 1 {
                    throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unsupported audio format: \(audioFormat) (only PCM supported)"])
                }
            } else if chunkId == "data" {
                dataStart = pos + 8
                dataSize = Int(chunkSize)
                break
            }
            
            pos += 8 + Int(chunkSize)
            if chunkSize % 2 == 1 { pos += 1 }
        }
        
        guard let sr = sampleRate, let ch = numChannels, let bps = bitsPerSample,
              let ds = dataStart, let dsz = dataSize else {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing fmt or data chunk in WAV file"])
        }
        
        // Validate format
        if ch != 1 || sr != 16000 || bps != 16 {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "WAV must be PCM16 mono @16kHz"])
        }
        
        // Determine effective data size: some recorders may leave data chunk size to 0 until finalization
        var effectiveDataSize = dsz
        if dsz == 0 || ds + dsz > data.count {
            effectiveDataSize = max(0, data.count - ds)
            NSLog("⚠️ [WHISPER] WAV data chunk reported size %d, using effective size %d", dsz, effectiveDataSize)
        }
        
        // Read PCM16 samples
        let bytesPerSample = bps / 8
        let totalSamples = effectiveDataSize / bytesPerSample / ch
        if totalSamples <= 0 {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "No audio samples in WAV data chunk"])
        }
        
        // Convert to float32
        var floatArray = [Float](repeating: 0, count: totalSamples)
        for i in 0..<totalSamples {
            let sampleOffset = ds + i * bytesPerSample * ch
            let sampleInt = data.withUnsafeBytes { bytes in
                Int16(bitPattern: UInt16(bytes.load(fromByteOffset: sampleOffset, as: UInt16.self).littleEndian))
            }
            floatArray[i] = Float(sampleInt) / 32768.0
        }
        
        return floatArray
    }
    
    private func transcribe(context: OpaquePointer, audioData: [Float]) throws -> String {
        // Use default Whisper parameters (like the fast example)
        let result = audioData.withUnsafeBufferPointer { buffer in
            whisper_full_bridge(context, UnsafeMutablePointer(mutating: buffer.baseAddress), Int32(buffer.count))
        }
        
        guard result == 0 else {
            let msg = "Whisper transcription failed (code \(result))"
            NSLog("❌ [WHISPER] %@", msg)
            throw NSError(domain: "WhisperPlugin", code: Int(result), userInfo: [NSLocalizedDescriptionKey: msg])
        }
        
        // Get result
        guard let resultPtr = whisper_get_result_bridge(context) else {
            throw NSError(domain: "WhisperPlugin", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to get transcription result"])
        }
        
        let transcript = String(cString: resultPtr)
        whisper_free_cstr_bridge(resultPtr)
        
        return transcript
    }
    
    deinit {
        if let context = cachedContext {
            whisper_free_bridge(context)
        }
    }
}
