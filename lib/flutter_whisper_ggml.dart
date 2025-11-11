import 'dart:io';

import 'whisper_method_channel.dart';

// Export implementations for direct access if needed
export 'whisper_method_channel.dart' show WhisperMethodChannel;

/// Main API for flutter_whisper_ggml
/// Automatically uses Method Channel on both Android and iOS
class Whisper {
  late final WhisperMethodChannel _impl;

  Whisper() {
    if (Platform.isAndroid || Platform.isIOS) {
      _impl = WhisperMethodChannel();
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

  Future<String> listenAndTranscribeSpeech({
    required String modelPath,
    required Stream<List<int>> audioStream,
  }) {
    return _impl.transcribeStream(
      modelPath: modelPath,
      audioStream: audioStream,
    );
  }
}
