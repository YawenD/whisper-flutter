import 'dart:io';

import 'whisper_ffi.dart';
import 'whisper_method_channel.dart';

// Export implementations for direct access if needed
export 'whisper_ffi.dart' show WhisperFFI;
export 'whisper_method_channel.dart' show WhisperMethodChannel;

/// Main API for flutter_whisper_ggml
/// Automatically uses Method Channel on Android and FFI on iOS
class Whisper {
  late final dynamic _impl;

  Whisper() {
    if (Platform.isAndroid) {
      _impl = WhisperMethodChannel();
    } else if (Platform.isIOS) {
      _impl = WhisperFFI();
    } else {
      throw UnsupportedError(
        'Platform ${Platform.operatingSystem} is not supported',
      );
    }
  }

  /// Copie le modèle depuis les assets si besoin
  Future<String> prepareModel(String existingModelPath) {
    return _impl.prepareModel(existingModelPath);
  }

  /// Transcrit un fichier WAV
  Future<String> transcribe({
    required String modelPath,
    required String wavPath,
  }) {
    return _impl.transcribe(modelPath: modelPath, wavPath: wavPath);
  }
}
