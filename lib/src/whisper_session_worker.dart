import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter_whisper_ggml/src/ios_whisper_ffi.dart';
import 'package:flutter_whisper_ggml/src/whisper_model.dart';
import 'package:flutter_whisper_ggml/src/whisper_stats.dart';
import 'package:flutter_whisper_ggml/utils.dart' show convertBytesToFloat32;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

/// Result of a transcription with its native performance/confidence figures.
class WhisperTranscription {
  const WhisperTranscription(this.text, this.stats);

  final String text;

  /// Null when nothing was transcribed (e.g. empty input).
  final WhisperStats? stats;
}

/// Raised when the worker isolate died or stopped answering.
class WhisperWorkerException implements Exception {
  WhisperWorkerException(this.message);

  final String message;

  @override
  String toString() => 'WhisperWorkerException: $message';
}

/// Runs a native Whisper session in a background isolate so CPU-bound
/// inference never blocks the UI isolate.
///
/// Requests are multiplexed over a single port with ids. If the isolate dies
/// (uncaught error, native abort reported as an isolate exit...), every
/// pending and future request fails with [WhisperWorkerException] instead of
/// hanging forever.
class WhisperSessionWorker {
  WhisperSessionWorker._(
    this._isolate,
    this._commandPort,
    this._responsePort,
    this._exitPort,
    Stream<dynamic> exitEvents,
    this._errorPort,
  ) {
    _responsePort.listen(_onResponse);
    exitEvents.listen((_) => _fail('Whisper worker isolate exited'));
    _errorPort.listen((dynamic error) {
      final description = error is List && error.isNotEmpty
          ? error.first
          : error;
      _fail('Whisper worker isolate crashed: $description');
    });
  }

  /// Upper bound for a single request; inference of a 30 s segment with a big
  /// model on a slow phone stays well below it.
  static const Duration requestTimeout = Duration(minutes: 2);

  final Isolate _isolate;
  final SendPort _commandPort;
  final ReceivePort _responsePort;
  final ReceivePort _exitPort;
  final ReceivePort _errorPort;
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;
  bool _isDisposed = false;
  String? _deadReason;

  /// Whether the worker can still accept requests.
  bool get isAlive => !_isDisposed && _deadReason == null;

  /// Start a whisper.cpp worker on the ggml model at [modelPath].
  static Future<WhisperSessionWorker> spawn(
    String modelPath, {
    String language = 'fr',
    WhisperDecodingOptions options = const WhisperDecodingOptions(),
  }) => spawnEngine(
    WhisperEngineSpec.ggml(modelPath),
    language: language,
    options: options,
  );

  /// Start a worker running the engine described by [spec] (see
  /// [WhisperModel.resolve]).
  static Future<WhisperSessionWorker> spawnEngine(
    WhisperEngineSpec spec, {
    String language = 'fr',
    WhisperDecodingOptions options = const WhisperDecodingOptions(),
  }) async {
    final handshakePort = ReceivePort();
    final responsePort = ReceivePort();
    final exitPort = ReceivePort();
    final errorPort = ReceivePort();

    final Isolate isolate;
    try {
      isolate = await Isolate.spawn(
        _WhisperSessionIsolate.entryPoint,
        <dynamic>[
          spec.toMap(),
          language,
          options.toMap(),
          handshakePort.sendPort,
          responsePort.sendPort,
        ],
        debugName: 'WhisperSessionWorker',
        onExit: exitPort.sendPort,
        onError: errorPort.sendPort,
      );
    } catch (_) {
      for (final port in [handshakePort, responsePort, exitPort, errorPort]) {
        port.close();
      }
      rethrow;
    }

    // The isolate answers with its command port, or with an error map when
    // the native session cannot be created (missing/corrupt model, unknown
    // language, unsupported CPU...). An exit without answer is also handled.
    // A ReceivePort supports a single listener: share the exit events between
    // the handshake and the worker.
    final exitEvents = exitPort.asBroadcastStream();
    final exitDuringHandshake = exitEvents.first.then((_) => null);
    final handshake = await Future.any<dynamic>([
      handshakePort.first,
      exitDuringHandshake,
    ]);
    handshakePort.close();

    if (handshake is! SendPort) {
      isolate.kill(priority: Isolate.immediate);
      for (final port in [responsePort, exitPort, errorPort]) {
        port.close();
      }
      final reason = handshake is Map
          ? handshake['error']
          : 'isolate exited during startup';
      throw WhisperWorkerException(
        'Unable to start the Whisper worker: $reason',
      );
    }

    return WhisperSessionWorker._(
      isolate,
      handshake,
      responsePort,
      exitPort,
      exitEvents,
      errorPort,
    );
  }

  Future<void> appendPcmBytes(Uint8List bytes) async {
    await _send(_Msg.appendPcm, {'bytes': bytes});
  }

  /// Transcribe [samples] (16 kHz mono float PCM) and return the text only.
  Future<String> transcribeFloatSamples(Float32List samples) async {
    return (await transcribe(samples)).text;
  }

  /// Transcribe [samples] (16 kHz mono float PCM) with native stats.
  Future<WhisperTranscription> transcribe(Float32List samples) async {
    final result = await _send(_Msg.transcribeFloat, {'samples': samples});
    return _decodeTranscription(result);
  }

  /// Transcribe the PCM16 audio appended with [appendPcmBytes] so far.
  Future<WhisperTranscription> transcribeBuffered() async {
    final result = await _send(_Msg.transcribeBuffered);
    return _decodeTranscription(result);
  }

  /// Apply decoding [options] to the following transcriptions.
  Future<void> configure(WhisperDecodingOptions options) async {
    await _send(_Msg.configure, {'options': options.toMap()});
  }

  /// Run one throwaway inference so the first real one does not pay the
  /// one-off allocations (and Metal pipeline compilation on iOS).
  Future<WhisperStats?> warmUp() async {
    final result = await _send(_Msg.warmUp);
    return _decodeTranscription(result).stats;
  }

  /// whisper.cpp / ggml build features as seen by the worker's native library.
  Future<String> systemInfo() async {
    final result = await _send(_Msg.systemInfo);
    return (result as String?) ?? '';
  }

  Future<void> reset() async {
    await _send(_Msg.reset);
  }

  Future<void> dispose() async {
    if (_isDisposed) {
      return;
    }
    if (_deadReason == null) {
      try {
        // Lets the isolate destroy the native session (releasing the model)
        // before it is killed.
        await _send(_Msg.dispose).timeout(const Duration(seconds: 5));
      } catch (_) {
        // The isolate is gone or stuck: killing it below is all we can do.
      }
    }
    _isDisposed = true;
    _isolate.kill(priority: Isolate.immediate);
    _fail('Worker disposed');
    _responsePort.close();
    _exitPort.close();
    _errorPort.close();
  }

  WhisperTranscription _decodeTranscription(dynamic result) {
    final map = result as Map;
    final stats = map['stats'] as Map?;
    return WhisperTranscription(
      (map['text'] as String?) ?? '',
      stats == null ? null : WhisperStats.fromMap(stats),
    );
  }

  Future<dynamic> _send(String type, [Map<String, dynamic>? payload]) {
    if (_isDisposed && type != _Msg.dispose) {
      return Future.error(WhisperWorkerException('Worker already disposed'));
    }
    final dead = _deadReason;
    if (dead != null) {
      return Future.error(WhisperWorkerException(dead));
    }

    final id = _nextId++;
    final completer = Completer<dynamic>();
    _pending[id] = completer;
    _commandPort.send(<String, dynamic>{...?payload, 'type': type, 'id': id});

    return completer.future.timeout(
      requestTimeout,
      onTimeout: () {
        _pending.remove(id);
        throw WhisperWorkerException('Request "$type" timed out');
      },
    );
  }

  void _onResponse(dynamic message) {
    if (message is! Map) return;
    final completer = _pending.remove(message['id']);
    if (completer == null) return;
    final error = message['error'];
    if (error != null) {
      completer.completeError(Exception(error as String));
    } else {
      completer.complete(message['result']);
    }
  }

  void _fail(String reason) {
    _deadReason ??= reason;
    final pending = _pending.values.toList();
    _pending.clear();
    for (final completer in pending) {
      if (!completer.isCompleted) {
        completer.completeError(WhisperWorkerException(reason));
      }
    }
  }
}

class _Msg {
  static const String appendPcm = 'appendPcm';
  static const String transcribeFloat = 'transcribeFloat';
  static const String transcribeBuffered = 'transcribeBuffered';
  static const String configure = 'configure';
  static const String warmUp = 'warmUp';
  static const String systemInfo = 'systemInfo';
  static const String reset = 'reset';
  static const String dispose = 'dispose';
}

/// An inference engine living inside the worker isolate.
abstract class _Engine {
  void appendPcm16(Uint8List bytes);
  void appendFloat(Float32List samples);

  /// Transcribe the appended audio. Does not clear it.
  String transcribe();
  WhisperStats? get lastStats;
  void configure(WhisperDecodingOptions options);
  void reset();
  void dispose();
  String get systemInfo;

  static _Engine create(
    WhisperEngineSpec spec,
    String language,
    WhisperDecodingOptions options,
  ) {
    switch (spec.engine) {
      case 'ggml':
        return _WhisperCppEngine(spec.files['model']!, language)
          ..configure(options);
      case WhisperEngineSpec.sherpaWhisperEngine:
      case WhisperEngineSpec.canaryEngine:
      case WhisperEngineSpec.transducerEngine:
      case WhisperEngineSpec.nemoCtcEngine:
        return _SherpaOnnxEngine(spec, language);
      default:
        throw ArgumentError('Unknown engine: ${spec.engine}');
    }
  }
}

class _WhisperCppEngine implements _Engine {
  _WhisperCppEngine(String modelPath, String language)
    : _session = WhisperStreamSession.create(modelPath, language: language);

  final WhisperStreamSession _session;

  @override
  void appendPcm16(Uint8List bytes) => _session.appendPcmBytes(bytes);

  @override
  void appendFloat(Float32List samples) => _session.appendFloatSamples(samples);

  @override
  String transcribe() => _session.transcribeSync().trim();

  @override
  WhisperStats? get lastStats => _session.lastStats;

  @override
  void configure(WhisperDecodingOptions options) => _session.configure(options);

  @override
  void reset() => _session.reset();

  @override
  void dispose() => _session.dispose();

  @override
  String get systemInfo => WhisperNativeFfi.instance.systemInfo;
}

/// Whisper exported to ONNX, run by sherpa-onnx (ONNX Runtime).
class _SherpaOnnxEngine implements _Engine {
  _SherpaOnnxEngine(WhisperEngineSpec spec, String language)
    : _threads = spec.threads {
    for (final file in spec.files.values) {
      if (!File(file).existsSync()) {
        throw ArgumentError.value(file, 'file', 'Model file not found');
      }
    }
    // Bindings are per isolate.
    sherpa.initBindings();
    _isNemo = spec.engine != WhisperEngineSpec.sherpaWhisperEngine;
    _recognizer = sherpa.OfflineRecognizer(
      sherpa.OfflineRecognizerConfig(model: _modelConfig(spec, language)),
    );
  }

  static sherpa.OfflineModelConfig _modelConfig(
    WhisperEngineSpec spec,
    String language,
  ) {
    final f = spec.files;
    switch (spec.engine) {
      case WhisperEngineSpec.canaryEngine:
        if (language == 'auto') {
          throw ArgumentError('Canary needs an explicit language, not "auto".');
        }
        return sherpa.OfflineModelConfig(
          canary: sherpa.OfflineCanaryModelConfig(
            encoder: f['encoder']!,
            decoder: f['decoder']!,
            srcLang: language,
            tgtLang: language,
          ),
          tokens: f['tokens']!,
          numThreads: spec.threads,
          debug: false,
        );
      case WhisperEngineSpec.transducerEngine:
        return sherpa.OfflineModelConfig(
          transducer: sherpa.OfflineTransducerModelConfig(
            encoder: f['encoder']!,
            decoder: f['decoder']!,
            joiner: f['joiner']!,
          ),
          tokens: f['tokens']!,
          modelType: 'nemo_transducer',
          numThreads: spec.threads,
          debug: false,
        );
      case WhisperEngineSpec.nemoCtcEngine:
        return sherpa.OfflineModelConfig(
          nemoCtc: sherpa.OfflineNemoEncDecCtcModelConfig(model: f['model']!),
          tokens: f['tokens']!,
          numThreads: spec.threads,
          debug: false,
        );
      default:
        return sherpa.OfflineModelConfig(
          whisper: sherpa.OfflineWhisperModelConfig(
            encoder: f['encoder']!,
            decoder: f['decoder']!,
            // Empty = automatic language detection.
            language: language == 'auto' ? '' : language,
            task: 'transcribe',
          ),
          tokens: f['tokens']!,
          numThreads: spec.threads,
          debug: false,
        );
    }
  }

  late final bool _isNemo;

  final int _threads;
  late final sherpa.OfflineRecognizer _recognizer;
  final List<Float32List> _chunks = [];
  int _sampleCount = 0;
  WhisperStats? _lastStats;

  @override
  void appendPcm16(Uint8List bytes) =>
      appendFloat(convertBytesToFloat32(bytes));

  @override
  void appendFloat(Float32List samples) {
    _chunks.add(Float32List.fromList(samples));
    _sampleCount += samples.length;
  }

  @override
  String transcribe() {
    if (_sampleCount == 0) {
      throw StateError('No audio to transcribe');
    }
    final samples = Float32List(_sampleCount);
    var offset = 0;
    for (final c in _chunks) {
      samples.setRange(offset, offset + c.length, c);
      offset += c.length;
    }

    final stream = _recognizer.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: 16000);
      final watch = Stopwatch()..start();
      _recognizer.decode(stream);
      watch.stop();
      final result = _recognizer.getResult(stream);
      _lastStats = WhisperStats(
        totalMs: watch.elapsedMicroseconds / 1000,
        sampleMs: 0,
        encodeMs: 0,
        decodeMs: 0,
        batchdMs: 0,
        promptMs: 0,
        sampleCount: _sampleCount,
        threadCount: _threads,
        audioCtx: 0,
        tokenCount: result.tokens.length,
        // Not exposed by sherpa-onnx: neutral values, so the no-speech rule
        // of HallucinationFilter never fires for this engine.
        noSpeechProb: 0,
        avgLogprob: 0,
      );
      final text = result.text.trim();
      return _isNemo ? normalizeNemoText(text) : text;
    } finally {
      stream.free();
    }
  }

  @override
  WhisperStats? get lastStats => _lastStats;

  @override
  void configure(WhisperDecodingOptions options) {
    // Initial prompt, dynamic audio context, fallback: not supported by the
    // sherpa-onnx Whisper decoder.
  }

  @override
  void reset() {
    _chunks.clear();
    _sampleCount = 0;
  }

  @override
  void dispose() {
    reset();
    _recognizer.free();
  }

  @override
  String get systemInfo => 'sherpa-onnx (ONNX Runtime, CPU, $_threads threads)';
}

/// NeMo tokenizers emit hyphenated words with spaces around the hyphen
/// ("Rappelle - moi", "va - t - il"): glue them back.
String normalizeNemoText(String text) =>
    text.replaceAll(RegExp(r'(?<=\S)\s+-\s+(?=\S)'), '-');

class _WhisperSessionIsolate {
  static void entryPoint(List<dynamic> args) async {
    final spec = WhisperEngineSpec.fromMap(args[0] as Map);
    final language = args[1] as String;
    final options = WhisperDecodingOptions.fromMap(args[2] as Map);
    final SendPort handshakePort = args[3] as SendPort;
    final SendPort responsePort = args[4] as SendPort;

    final _Engine session;
    try {
      session = _Engine.create(spec, language, options);
    } on Object catch (error) {
      handshakePort.send(<String, dynamic>{'error': error.toString()});
      return;
    }

    final commandPort = ReceivePort();
    handshakePort.send(commandPort.sendPort);

    Map<String, Object?> transcription(String text, WhisperStats? stats) => {
      'text': text,
      'stats': stats?.toMap(),
    };

    await for (final dynamic raw in commandPort) {
      if (raw is! Map) {
        continue;
      }
      final type = raw['type'] as String?;
      final id = raw['id'];

      void respond(Object? result) =>
          responsePort.send(<String, Object?>{'id': id, 'result': result});

      try {
        switch (type) {
          case _Msg.appendPcm:
            final bytes = raw['bytes'] as Uint8List? ?? Uint8List(0);
            session.appendPcm16(bytes);
            respond(null);
          case _Msg.transcribeFloat:
            final samples = raw['samples'] as Float32List? ?? Float32List(0);
            if (samples.isEmpty) {
              respond(transcription('', null));
              break;
            }
            try {
              session.appendFloat(samples);
              final text = session.transcribe();
              respond(transcription(text, session.lastStats));
            } finally {
              session.reset();
            }
          case _Msg.transcribeBuffered:
            try {
              final text = session.transcribe();
              respond(transcription(text, session.lastStats));
            } finally {
              session.reset();
            }
          case _Msg.warmUp:
            try {
              // One second of near-silence (exact zeros can short-circuit
              // some kernels) exercises the whole encoder/decoder path.
              final samples = Float32List(16000);
              for (var i = 0; i < samples.length; i += 1) {
                samples[i] = ((i * 7919) % 200 - 100) / 1e5;
              }
              session.appendFloat(samples);
              session.transcribe();
              respond(transcription('', session.lastStats));
            } finally {
              session.reset();
            }
          case _Msg.configure:
            session.configure(
              WhisperDecodingOptions.fromMap(raw['options'] as Map),
            );
            respond(null);
          case _Msg.systemInfo:
            respond(session.systemInfo);
          case _Msg.reset:
            session.reset();
            respond(null);
          case _Msg.dispose:
            session.dispose();
            respond(null);
            commandPort.close();
            return;
          default:
            responsePort.send(<String, Object?>{
              'id': id,
              'error': 'Unknown worker message: $type',
            });
        }
      } on Object catch (error, stackTrace) {
        if (kDebugMode) {
          debugPrint('[WhisperSessionWorker] Error: $error\n$stackTrace');
        }
        responsePort.send(<String, Object?>{
          'id': id,
          'error': error.toString(),
        });
      }
    }
  }
}
