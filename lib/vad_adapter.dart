import 'dart:typed_data';

import 'package:flutter_whisper_ggml/utils.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

// Export implementations for direct access if needed
export 'whisper_method_channel.dart' show WhisperMethodChannel;

/// Tuning of the Silero voice activity detector.
class VadSettings {
  const VadSettings({
    this.threshold = 0.5,
    this.minSilenceDuration = 0.8,
    this.minSpeechDuration = 0.25,
    this.maxSpeechDuration = 20.0,
  });

  /// Speech probability above which a window counts as speech.
  final double threshold;

  /// Silence (seconds) that ends a speech segment. This is a floor on the
  /// latency between the end of a sentence and its transcript: lower is more
  /// reactive, but natural pauses may split sentences.
  final double minSilenceDuration;

  /// Shorter bursts (seconds) are ignored (clicks, breaths...).
  final double minSpeechDuration;

  /// Longer speech (seconds) is cut into several segments, which bounds the
  /// latency when someone talks without pausing (Whisper's window is 30 s).
  final double maxSpeechDuration;
}

/// A completed speech segment.
class VadSegment {
  const VadSegment(this.samples, this.start);

  final Float32List samples;

  /// Position of the first sample, counted in samples fed to the detector
  /// since its last [IVadAdapter.reset] (or [IVadAdapter.flush]).
  final int start;

  int get end => start + samples.length;
}

abstract class IVadAdapter {
  Future<void> initSilero({required String modelPath});

  /// Feed PCM16 little-endian mono 16 kHz bytes and return the speech
  /// segments completed by this chunk (usually none).
  List<VadSegment> acceptPcm16(Uint8List bytes);

  /// Force the segment in progress out of the detector (e.g. when the user
  /// stops while still speaking) and return the remaining speech segments.
  List<VadSegment> flush();

  /// Audio of the speech segment currently in progress (with a short pre-roll),
  /// or null when nobody is speaking. Used for partial transcripts.
  Float32List? get currentSpeech;

  /// Position of the first sample of [currentSpeech], on the same scale as
  /// [VadSegment.start]; null when nobody is speaking.
  int? get currentSpeechStart;

  /// Drop any buffered audio and the detector state (new session, pause...).
  void reset();

  Future<void> dispose();
}

class VadAdapter implements IVadAdapter {
  VadAdapter({this.settings = const VadSettings()});

  static const int _sampleRate = 16000;

  final VadSettings settings;

  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.CircularBuffer? _buffer;
  late sherpa_onnx.VadModelConfig _vadConfig;

  // Rolling pre-roll kept while nobody speaks: Silero only flags speech after
  // a few windows, so the in-progress buffer starts a bit earlier.
  final List<Float32List> _preRoll = [];
  int _preRollSamples = 0;
  final List<Float32List> _speech = [];
  int _speechStart = 0;
  bool _wasDetected = false;

  /// Samples fed to the detector since the last reset.
  int _fed = 0;

  int get _preRollCapacity =>
      ((settings.minSpeechDuration + 0.3) * _sampleRate).round();

  @override
  Future<void> initSilero({required String modelPath}) async {
    sherpa_onnx.initBindings();

    final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(
      model: await copyAssetFile(modelPath),
      threshold: settings.threshold,
      minSilenceDuration: settings.minSilenceDuration,
      minSpeechDuration: settings.minSpeechDuration,
      maxSpeechDuration: settings.maxSpeechDuration,
    );

    _vadConfig = sherpa_onnx.VadModelConfig(
      sileroVad: sileroVadConfig,
      numThreads: 1,
      debug: false,
    );

    _vad = sherpa_onnx.VoiceActivityDetector(
      config: _vadConfig,
      bufferSizeInSeconds: 30,
    );
    _buffer = sherpa_onnx.CircularBuffer(capacity: 30 * _sampleRate);
  }

  @override
  List<VadSegment> acceptPcm16(Uint8List bytes) {
    final vad = _vad;
    final buffer = _buffer;
    if (vad == null || buffer == null) {
      return const [];
    }
    buffer.push(convertBytesToFloat32(bytes));

    final segments = <VadSegment>[];
    final windowSize = _vadConfig.sileroVad.windowSize;
    while (buffer.size >= windowSize) {
      final window = buffer.get(startIndex: buffer.head, n: windowSize);
      buffer.pop(windowSize);
      vad.acceptWaveform(window);
      _trackSpeech(window, vad.isDetected());
      _drainSegments(segments);
    }
    return segments;
  }

  @override
  List<VadSegment> flush() {
    final vad = _vad;
    final buffer = _buffer;
    if (vad == null || buffer == null) {
      return const [];
    }

    // Feed the leftover samples smaller than a full VAD window.
    if (buffer.size > 0) {
      final remaining = buffer.get(startIndex: buffer.head, n: buffer.size);
      buffer.pop(buffer.size);
      vad.acceptWaveform(remaining);
    }

    vad.flush();
    final segments = <VadSegment>[];
    _drainSegments(segments);
    // flush() finalizes the stream: start the next session from a clean state.
    reset();
    return segments;
  }

  @override
  Float32List? get currentSpeech {
    if (_speech.isEmpty) {
      return null;
    }
    return _concat(_speech);
  }

  @override
  int? get currentSpeechStart => _speech.isEmpty ? null : _speechStart;

  @override
  void reset() {
    _vad?.reset();
    _buffer?.reset();
    _preRoll.clear();
    _preRollSamples = 0;
    _speech.clear();
    _wasDetected = false;
    _fed = 0;
  }

  void _trackSpeech(Float32List window, bool detected) {
    if (detected) {
      if (!_wasDetected) {
        _speechStart = _fed - _preRollSamples;
        _speech
          ..clear()
          ..addAll(_preRoll);
        _preRoll.clear();
        _preRollSamples = 0;
      }
      _speech.add(window);
    } else {
      _speech.clear();
      _preRoll.add(window);
      _preRollSamples += window.length;
      while (_preRollSamples - _preRoll.first.length >= _preRollCapacity) {
        _preRollSamples -= _preRoll.removeAt(0).length;
      }
    }
    _wasDetected = detected;
    _fed += window.length;
  }

  void _drainSegments(List<VadSegment> out) {
    final vad = _vad!;
    while (!vad.isEmpty()) {
      final segment = vad.front();
      vad.pop();
      out.add(VadSegment(segment.samples, segment.start));
      // The in-progress speech became a final segment.
      _speech.clear();
    }
  }

  static Float32List _concat(List<Float32List> parts) {
    final total = parts.fold<int>(0, (n, p) => n + p.length);
    final out = Float32List(total);
    var offset = 0;
    for (final p in parts) {
      out.setRange(offset, offset + p.length, p);
      offset += p.length;
    }
    return out;
  }

  @override
  Future<void> dispose() async {
    _vad?.free();
    _buffer?.free();
    _vad = null;
    _buffer = null;
  }
}
