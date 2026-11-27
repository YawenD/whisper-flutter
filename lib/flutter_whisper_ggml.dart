import 'dart:io';

import 'whisper_method_channel.dart';

export 'live_transcriber.dart'
    show WhisperLiveTranscriber, WhisperBacklogException, TimedTranscript;
export 'src/hallucination_filter.dart' show HallucinationFilter;
export 'src/whisper_session_worker.dart'
    show WhisperSessionWorker, WhisperTranscription, WhisperWorkerException;
export 'src/whisper_stats.dart' show WhisperStats, WhisperDecodingOptions;
export 'src/whisper_model.dart'
    show
        WhisperModel,
        GgmlWhisperModel,
        SherpaOnnxWhisperModel,
        SherpaOnnxCanaryModel,
        SherpaOnnxTransducerModel,
        SherpaOnnxNemoCtcModel,
        WhisperEngineSpec;
export 'vad_adapter.dart' show VadAdapter, VadSettings, IVadAdapter, VadSegment;
export 'record_adapter.dart' show RecordAdapter, IRecordAdapter;
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

  /// Copy the model from the assets when needed.
  Future<String> prepareModel(String existingModelPath) {
    return _impl.prepareModel(existingModelPath);
  }

  /// Transcribe a WAV file.
  Future<String> transcribe({
    required String modelPath,
    required String wavPath,
  }) {
    return _impl.transcribe(modelPath: modelPath, wavPath: wavPath);
  }
}
