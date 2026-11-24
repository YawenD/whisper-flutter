/// Performance and confidence figures of one native transcription.
///
/// `encodeMs`, `decodeMs`, ... are the per-pass averages reported by
/// whisper.cpp (`decodeMs` is per token); `totalMs` is the wall-clock time of
/// the whole `whisper_full` call.
class WhisperStats {
  const WhisperStats({
    required this.totalMs,
    required this.sampleMs,
    required this.encodeMs,
    required this.decodeMs,
    required this.batchdMs,
    required this.promptMs,
    required this.sampleCount,
    required this.threadCount,
    required this.audioCtx,
    required this.tokenCount,
    required this.noSpeechProb,
    required this.avgLogprob,
  });

  final double totalMs;
  final double sampleMs;
  final double encodeMs;
  final double decodeMs;
  final double batchdMs;
  final double promptMs;
  final int sampleCount;
  final int threadCount;

  /// Encoder context used (1500 = full 30 s window).
  final int audioCtx;

  /// Decoded text tokens (special tokens excluded).
  final int tokenCount;

  /// Highest probability, over the segments, that the audio holds no speech.
  final double noSpeechProb;

  /// Mean log-probability of the decoded text tokens (closer to 0 = surer).
  final double avgLogprob;

  /// Duration of the transcribed audio, in milliseconds (16 kHz input).
  double get audioMs => sampleCount / 16.0;

  /// Real-time factor: processing time / audio duration (< 1 is faster than
  /// real time).
  double get realTimeFactor => audioMs == 0 ? 0 : totalMs / audioMs;

  Map<String, Object> toMap() => {
    'totalMs': totalMs,
    'sampleMs': sampleMs,
    'encodeMs': encodeMs,
    'decodeMs': decodeMs,
    'batchdMs': batchdMs,
    'promptMs': promptMs,
    'sampleCount': sampleCount,
    'threadCount': threadCount,
    'audioCtx': audioCtx,
    'tokenCount': tokenCount,
    'noSpeechProb': noSpeechProb,
    'avgLogprob': avgLogprob,
  };

  static WhisperStats fromMap(Map<dynamic, dynamic> map) => WhisperStats(
    totalMs: map['totalMs'] as double,
    sampleMs: map['sampleMs'] as double,
    encodeMs: map['encodeMs'] as double,
    decodeMs: map['decodeMs'] as double,
    batchdMs: map['batchdMs'] as double,
    promptMs: map['promptMs'] as double,
    sampleCount: map['sampleCount'] as int,
    threadCount: map['threadCount'] as int,
    audioCtx: map['audioCtx'] as int,
    tokenCount: map['tokenCount'] as int,
    noSpeechProb: map['noSpeechProb'] as double,
    avgLogprob: map['avgLogprob'] as double,
  );

  @override
  String toString() =>
      'total=${totalMs.toStringAsFixed(0)}ms '
      'audio=${audioMs.toStringAsFixed(0)}ms '
      'rtf=${realTimeFactor.toStringAsFixed(2)} '
      'encode=${encodeMs.toStringAsFixed(0)}ms '
      'decode=${decodeMs.toStringAsFixed(1)}ms/tok '
      'ctx=$audioCtx tokens=$tokenCount '
      'nosp=${noSpeechProb.toStringAsFixed(2)} '
      'logp=${avgLogprob.toStringAsFixed(2)} '
      'threads=$threadCount';
}

/// Decoding options applied to a native session.
class WhisperDecodingOptions {
  const WhisperDecodingOptions({
    this.dynamicAudioCtx = false,
    this.initialPrompt,
    this.audioCtxMargin = 0,
    this.temperatureInc = 0,
    this.bestOf = 1,
    this.threads = 0,
  });

  /// Shrink the encoder context to the segment length. 2-6x faster on short
  /// phrases, but measured to badly hurt accuracy in French (mean WER ~60-80%
  /// vs ~33% with the full window on base-q5_1, with repetition loops):
  /// Whisper was trained on 30 s windows. Off by default; experimental.
  final bool dynamicAudioCtx;

  /// Text given to the decoder as context: domain vocabulary, proper nouns,
  /// spelling conventions... Keep it short (a sentence or a word list).
  final String? initialPrompt;

  /// Encoder frames (50 per second) kept after the audio when
  /// [dynamicAudioCtx] is on; 0 uses the native default. A tight context is
  /// faster but makes small models more likely to loop or hallucinate.
  final int audioCtxMargin;

  /// Temperature step of the fallback re-decoding triggered when a decode
  /// looks unreliable (repetition loop, low confidence). 0 disables it
  /// (lowest, steadiest latency); 0.4 allows up to 2 retries.
  final double temperatureInc;

  /// Candidates sampled per fallback retry.
  final int bestOf;

  /// CPU threads used by the inference; 0 keeps the platform default (on
  /// Android: the number of big cores, at least 2).
  final int threads;

  Map<String, Object?> toMap() => {
    'dynamicAudioCtx': dynamicAudioCtx,
    'initialPrompt': initialPrompt,
    'audioCtxMargin': audioCtxMargin,
    'temperatureInc': temperatureInc,
    'bestOf': bestOf,
    'threads': threads,
  };

  static WhisperDecodingOptions fromMap(Map<dynamic, dynamic> map) =>
      WhisperDecodingOptions(
        dynamicAudioCtx: map['dynamicAudioCtx'] as bool? ?? false,
        initialPrompt: map['initialPrompt'] as String?,
        audioCtxMargin: map['audioCtxMargin'] as int? ?? 0,
        temperatureInc: map['temperatureInc'] as double? ?? 0,
        bestOf: map['bestOf'] as int? ?? 1,
        threads: map['threads'] as int? ?? 0,
      );
}
