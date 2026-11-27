import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_whisper_ggml/record_adapter.dart';
import 'package:flutter_whisper_ggml/src/hallucination_filter.dart';
import 'package:flutter_whisper_ggml/src/speech_queue.dart';
import 'package:flutter_whisper_ggml/src/whisper_model.dart';
import 'package:flutter_whisper_ggml/src/whisper_session_worker.dart';
import 'package:flutter_whisper_ggml/src/whisper_stats.dart';
import 'package:flutter_whisper_ggml/vad_adapter.dart';

/// Raised on the transcripts stream when inference cannot keep up and old
/// speech segments had to be dropped.
class WhisperBacklogException implements Exception {
  WhisperBacklogException(this.droppedSeconds);

  final double droppedSeconds;

  @override
  String toString() =>
      'WhisperBacklogException: inference too slow, dropped '
      '${droppedSeconds.toStringAsFixed(1)} s of speech';
}

/// A transcript with the span of captured audio it covers, counted from the
/// start of listening (pauses of the microphone stream excluded, since no
/// audio arrives then).
class TimedTranscript {
  const TimedTranscript(this.text, this.start, this.end);

  final String text;
  final Duration start;
  final Duration end;

  @override
  String toString() => 'TimedTranscript($start-$end, "$text")';
}

enum _State { idle, starting, listening, stopping }

/// Live speech-to-text: microphone → Silero VAD → Whisper (background
/// isolate) → [transcripts].
///
/// Pipeline:
/// - every audio chunk goes through the VAD as soon as it arrives, even while
///   Whisper is busy, so detection never lags behind;
/// - completed speech segments go to a queue consumed by a single inference
///   loop; when several are waiting they are merged into one inference (the
///   encoder has a fixed cost per call), and the queue is bounded;
/// - the model stays loaded between [startListening]/[stopListening] calls and
///   is warmed up once, so starting is instant; [dispose] releases it.
class WhisperLiveTranscriber {
  WhisperLiveTranscriber({
    WhisperMethodChannel? whisper,
    IRecordAdapter? recordAdapter,
    IVadAdapter? vadAdapter,
    this.language = 'fr',
    this.decodingOptions = const WhisperDecodingOptions(),
    this.hallucinationFilter = const HallucinationFilter(),
    this.enablePartialResults = false,
    this.partialInterval = const Duration(milliseconds: 1200),
    this.maxBacklog = const Duration(seconds: 60),
    this.onTranscription,
  }) : _methodChannel = whisper ?? WhisperMethodChannel(),
       _recordAdapter = recordAdapter ?? RecordAdapter(),
       _vadAdapter = vadAdapter ?? VadAdapter() {
    if (!_languagePattern.hasMatch(language)) {
      throw ArgumentError.value(
        language,
        'language',
        'Expected a whisper language code such as "fr", "en" or "auto"',
      );
    }
  }

  static final RegExp _languagePattern = RegExp(r'^(auto|[a-z]{2,3})$');
  static const int _sampleRate = 16000;

  /// Merged inferences stay below Whisper's 30 s window.
  static const int _maxBatchSamples = 28 * _sampleRate;

  /// Silence inserted between merged segments so Whisper sees the boundary.
  static const int _mergeGapSamples = _sampleRate ~/ 5;

  final WhisperMethodChannel _methodChannel;
  final IRecordAdapter _recordAdapter;
  final IVadAdapter _vadAdapter;

  /// Whisper language code ('fr', 'en', ..., or 'auto').
  final String language;

  /// Native decoding options (initial prompt, dynamic audio context).
  final WhisperDecodingOptions decodingOptions;

  /// Applied to final and partial transcripts.
  final HallucinationFilter hallucinationFilter;

  /// Emit provisional transcripts of the sentence being spoken on
  /// [partialTranscripts]. Costs extra inferences: enable on fast devices.
  final bool enablePartialResults;

  /// Minimum delay between two partial transcriptions.
  final Duration partialInterval;

  /// Queued speech beyond this is dropped (oldest first) and reported as a
  /// [WhisperBacklogException].
  final Duration maxBacklog;

  /// Diagnostics hook called for every final inference, including the
  /// transcripts dropped by the [hallucinationFilter].
  final void Function(WhisperTranscription result, bool dropped)?
  onTranscription;

  final StreamController<String> _transcripts =
      StreamController<String>.broadcast();
  final StreamController<String> _partials =
      StreamController<String>.broadcast();
  final StreamController<TimedTranscript> _timedTranscripts =
      StreamController<TimedTranscript>.broadcast();
  final StreamController<TimedTranscript> _timedPartials =
      StreamController<TimedTranscript>.broadcast();

  WhisperSessionWorker? _worker;
  StreamSubscription<Uint8List>? _recordSubscription;
  _State _state = _State.idle;
  Future<void> _lifecycle = Future.value();
  bool _vadReady = false;
  bool _captureEnabled = true;
  bool _disposed = false;
  WhisperEngineSpec? _engineSpec;

  late final SpeechQueue _queue = SpeechQueue(
    maxBatchSamples: _maxBatchSamples,
    mergeGapSamples: _mergeGapSamples,
    maxBacklogSamples: maxBacklog.inMilliseconds * _sampleRate ~/ 1000,
  );

  /// Samples received from the microphone since listening started.
  int _clock = 0;

  /// [_clock] when the VAD received its first sample since its last reset:
  /// VAD positions are relative to it. Null until then.
  int? _vadOrigin;
  Future<void>? _pump;
  Future<void>? _partialInFlight;
  DateTime _lastPartial = DateTime.fromMillisecondsSinceEpoch(0);
  int _utterance = 0;

  /// Final transcripts, one per speech segment (or merged segments).
  /// Broadcast stream, valid for the whole life of the transcriber.
  Stream<String> get transcripts => _transcripts.stream;

  /// Provisional transcripts of the sentence in progress (only when
  /// [enablePartialResults]). Each one supersedes the previous; the next
  /// final transcript supersedes them all.
  Stream<String> get partialTranscripts => _partials.stream;

  /// Same as [transcripts], with the span of audio each one covers.
  Stream<TimedTranscript> get timedTranscripts => _timedTranscripts.stream;

  /// Same as [partialTranscripts], with the span of the speech in progress.
  Stream<TimedTranscript> get timedPartialTranscripts => _timedPartials.stream;

  bool get isListening => _state == _State.listening;

  /// Make the Whisper model available and, when [modelPath] (the Silero VAD
  /// model asset) is given, initialize the VAD. Without VAD, the whole
  /// recording is transcribed at [stopListening].
  ///
  /// Pass either [model] (engine + files, e.g. [SherpaOnnxWhisperModel] on
  /// Android) or [modelAssetPath], a ggml model for whisper.cpp ([useCoreMl]:
  /// see [WhisperMethodChannel.prepareModel]). Calling it again with another
  /// model reloads the worker at the next start.
  Future<void> prepare({
    String? modelAssetPath,
    WhisperModel? model,
    String? modelPath,
    bool useCoreMl = true,
  }) async {
    if ((model == null) == (modelAssetPath == null)) {
      throw ArgumentError('Pass exactly one of model or modelAssetPath.');
    }
    final source =
        model ?? GgmlWhisperModel(modelAssetPath!, useCoreMl: useCoreMl);
    // ggml models go through the injected method channel (testability).
    final spec = source is GgmlWhisperModel
        ? WhisperEngineSpec.ggml(
            await _methodChannel.prepareModel(
              source.model,
              useCoreMl: source.useCoreMl,
            ),
          )
        : await source.resolve();
    if (_engineSpec != null && _engineSpec.toString() != spec.toString()) {
      await _serialized(() async {
        await _worker?.dispose();
        _worker = null;
      });
    }
    _engineSpec = spec;
    if (modelPath != null && !_vadReady) {
      await _vadAdapter.initSilero(modelPath: modelPath);
      _vadReady = true;
    }
  }

  /// Load the model in the background isolate and warm it up. Optional:
  /// [startListening] does it when needed, but calling it ahead (e.g. on
  /// screen open) makes the first start instant.
  Future<void> initSession() => _serialized(_ensureWorker);

  Future<void> startListening() => _serialized(() async {
    if (_disposed) throw StateError('Transcriber disposed.');
    if (_state != _State.idle) {
      throw StateError('Live transcription already started.');
    }
    _state = _State.starting;
    try {
      final worker = await _ensureWorker();
      await worker.reset();
      _vadAdapter.reset();
      _vadOrigin = null;
      _queue.clear();
      _clock = 0;
      _captureEnabled = true;

      final audio = await _recordAdapter.startStream();
      _recordSubscription = audio.listen(
        _onAudio,
        onError: (Object error, StackTrace stackTrace) {
          _transcripts.addError(error, stackTrace);
        },
        onDone: _onCaptureDone,
        cancelOnError: false,
      );
      _state = _State.listening;
    } catch (_) {
      _state = _State.idle;
      rethrow;
    }
  });

  /// Stop capturing, transcribe the speech still in progress and wait until
  /// every pending transcript has been emitted. The model stays loaded.
  Future<void> stopListening() => _serialized(_stop);

  /// Temporarily ignore the microphone (e.g. while the app speaks). Speech in
  /// progress is discarded; already detected segments are still transcribed.
  Future<void> pauseCapture() async {
    if (_state != _State.listening) return;
    _captureEnabled = false;
    if (_vadReady) {
      _vadAdapter.reset();
      _vadOrigin = null;
    } else {
      await _worker?.reset();
    }
  }

  Future<void> resumeCapture() async {
    if (_state != _State.listening) return;
    _captureEnabled = true;
  }

  /// Stop listening and release the model, the recorder and the VAD.
  Future<void> dispose() => _serialized(() async {
    if (_disposed) return;
    await _stop();
    _disposed = true;
    await _worker?.dispose();
    _worker = null;
    await _recordAdapter.dispose();
    await _vadAdapter.dispose();
    await _transcripts.close();
    await _partials.close();
    await _timedTranscripts.close();
    await _timedPartials.close();
  });

  // --- internals -----------------------------------------------------------

  /// Runs lifecycle operations one at a time, in call order.
  Future<T> _serialized<T>(Future<T> Function() op) {
    final result = _lifecycle.then((_) => op());
    _lifecycle = result.then<void>((_) {}, onError: (_) {});
    return result;
  }

  Future<WhisperSessionWorker> _ensureWorker() async {
    final existing = _worker;
    if (existing != null && existing.isAlive) {
      return existing;
    }
    await existing?.dispose();
    _worker = null;

    final spec = _engineSpec;
    if (spec == null) {
      throw StateError('prepare() must be called before starting a session.');
    }
    final worker = await WhisperSessionWorker.spawnEngine(
      spec,
      language: language,
      options: decodingOptions,
    );
    if (kDebugMode) {
      debugPrint('[WhisperLiveTranscriber] engine: $spec');
    }
    try {
      final stats = await worker.warmUp();
      if (kDebugMode) {
        debugPrint('[WhisperLiveTranscriber] warm-up: $stats');
      }
    } catch (error) {
      await worker.dispose();
      rethrow;
    }
    _worker = worker;
    return worker;
  }

  Future<void> _stop() async {
    if (_state != _State.listening) return;
    _state = _State.stopping;
    _captureEnabled = false;

    await _recordSubscription?.cancel();
    _recordSubscription = null;
    try {
      await _recordAdapter.stopStream();
    } catch (error, stackTrace) {
      _transcripts.addError(error, stackTrace);
    }

    if (_vadReady) {
      // Force out the speech in progress so the last words are not dropped.
      final origin = _vadOrigin ?? _clock;
      for (final segment in _vadAdapter.flush()) {
        _enqueue(segment, origin);
      }
      // flush() resets the detector.
      _vadOrigin = null;
      await _drain();
    } else {
      await _transcribeBuffered();
    }
    _state = _State.idle;
  }

  /// The recorder stream ended on its own (microphone taken by a call,
  /// permission revoked, audio session interrupted...).
  void _onCaptureDone() {
    if (_state != _State.listening) return;
    _transcripts.addError(
      StateError('Audio capture ended unexpectedly'),
      StackTrace.current,
    );
    unawaited(stopListening().catchError((Object _) {}));
  }

  void _onAudio(Uint8List chunk) {
    if (_state != _State.listening) return;
    final at = _clock;
    // Every received sample moves the clock, ignored or not, so positions
    // match the captured audio.
    _clock += chunk.length ~/ 2;
    if (!_captureEnabled) return;

    if (!_vadReady) {
      // No VAD: accumulate everything natively, transcribed at stop.
      _worker?.appendPcmBytes(chunk).catchError((Object error) {
        _transcripts.addError(error);
      });
      return;
    }

    // VAD runs synchronously on every chunk, whatever Whisper is doing.
    final origin = _vadOrigin ??= at;
    final segments = _vadAdapter.acceptPcm16(chunk);
    if (segments.isNotEmpty) {
      for (final segment in segments) {
        _enqueue(segment, origin);
      }
    } else if (enablePartialResults) {
      _maybeTranscribePartial();
    }
  }

  void _enqueue(VadSegment segment, int origin) {
    if (segment.samples.isEmpty) return;
    _utterance++;
    final dropped = _queue.add(
      QueuedSpeech(segment.samples, origin + segment.start),
    );
    if (dropped > 0) {
      _transcripts.addError(WhisperBacklogException(dropped / _sampleRate));
    }

    _pump ??= _runPump();
  }

  Future<void> _runPump() async {
    try {
      // Let a partial inference finish first: the worker is sequential anyway.
      await _partialInFlight;
      while (_queue.isNotEmpty) {
        await _transcribeFinal(_queue.take());
      }
    } finally {
      // Cleared in the same synchronous step as the final empty-queue check,
      // so a segment enqueued right after always starts a new pump.
      _pump = null;
    }
  }

  Future<void> _drain() async {
    while (_pump != null || _partialInFlight != null) {
      await _pump;
      await _partialInFlight;
    }
  }

  Future<void> _transcribeFinal(QueuedSpeech speech) async {
    final worker = _worker;
    if (worker == null) return;
    try {
      final result = await worker.transcribe(speech.samples);
      _emitFinal(result, speech.start, speech.end);
    } catch (error, stackTrace) {
      _transcripts.addError(error, stackTrace);
    }
  }

  Future<void> _transcribeBuffered() async {
    final worker = _worker;
    if (worker == null) return;
    try {
      // Without VAD the whole capture is one transcript.
      _emitFinal(await worker.transcribeBuffered(), 0, _clock);
    } catch (error, stackTrace) {
      _transcripts.addError(error, stackTrace);
    }
  }

  void _emitFinal(WhisperTranscription result, int start, int end) {
    final dropped = hallucinationFilter.isHallucination(
      result.text,
      result.stats,
    );
    onTranscription?.call(result, dropped);
    if (kDebugMode) {
      debugPrint(
        '[WhisperLiveTranscriber] ${dropped ? 'DROPPED ' : ''}'
        '${result.stats} "${result.text}"',
      );
    }
    if (!dropped && !_transcripts.isClosed) {
      _transcripts.add(result.text);
      _timedTranscripts.add(_timed(result.text, start, end));
    }
  }

  void _maybeTranscribePartial() {
    if (_pump != null || _partialInFlight != null) return;
    final now = DateTime.now();
    if (now.difference(_lastPartial) < partialInterval) return;
    final speech = _vadAdapter.currentSpeech;
    final speechStart = _vadAdapter.currentSpeechStart;
    final origin = _vadOrigin;
    if (speech == null || speechStart == null || origin == null) return;
    if (speech.length < _sampleRate) return;
    final start = origin + speechStart;
    final worker = _worker;
    if (worker == null) return;

    _lastPartial = now;
    final utterance = _utterance;
    _partialInFlight = () async {
      try {
        final result = await worker.transcribe(speech);
        // Ignore it when the sentence ended meanwhile (the final wins).
        if (utterance == _utterance &&
            !_partials.isClosed &&
            !hallucinationFilter.isHallucination(result.text, result.stats)) {
          _partials.add(result.text);
          _timedPartials.add(_timed(result.text, start, start + speech.length));
        }
      } catch (_) {
        // Partial results are best effort.
      } finally {
        _partialInFlight = null;
      }
    }();
  }

  static TimedTranscript _timed(String text, int start, int end) =>
      TimedTranscript(text, _duration(start), _duration(end));

  static Duration _duration(int samples) =>
      Duration(microseconds: samples * 1000000 ~/ _sampleRate);
}
