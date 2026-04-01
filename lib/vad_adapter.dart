import 'dart:typed_data';

import 'package:flutter_whisper_ggml/utils.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

// Export implementations for direct access if needed
export 'whisper_method_channel.dart' show WhisperMethodChannel;

abstract class IVadAdapter {
  Future<void> initSilero({required String modelPath});
  Stream<List<double>> listenToSpeechEnd(List<int> samples);
  Future<void> dispose();
}

class VadAdapter implements IVadAdapter {
  static const int _sampleRate = 16000;

  // VAD related vars
  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.CircularBuffer? _buffer;

  // VAD config
  late sherpa_onnx.VadModelConfig _vadConfig;

  @override
  Future<void> initSilero({required String modelPath}) async {
    sherpa_onnx.initBindings();

    final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(
      model: await copyAssetFile(modelPath),
      // threshold: 0.6,
      minSilenceDuration: 1.75,
      minSpeechDuration: 0.3,
      maxSpeechDuration: 60.0,
    );

    _vadConfig = sherpa_onnx.VadModelConfig(
      sileroVad: sileroVadConfig,
      numThreads: 1,
      debug: false,
    );

    // create VAD, use buffer model
    _vad = sherpa_onnx.VoiceActivityDetector(
      config: _vadConfig,
      bufferSizeInSeconds: 30,
    );
    _buffer = sherpa_onnx.CircularBuffer(capacity: 30 * _sampleRate);
  }

  @override
  Stream<List<double>> listenToSpeechEnd(List<int> samples) async* {
    final samplesFloat32 = convertBytesToFloat32(Uint8List.fromList(samples));
    _buffer?.push(samplesFloat32);

    final windowSize = _vadConfig.sileroVad.windowSize;
    while (_buffer!.size > windowSize) {
      final samples = _buffer!.get(startIndex: _buffer!.head, n: windowSize);
      _buffer!.pop(windowSize);
      _vad!.acceptWaveform(samples);

      while (!_vad!.isEmpty()) {
        final segment = _vad!.front();
        _vad!.pop();
        final samples = segment.samples;
        yield samples;
      }
    }
  }

  @override
  Future<void> dispose() async {
    _vad?.free(); // release vad
    _buffer?.free(); // release buffer
  }
}
