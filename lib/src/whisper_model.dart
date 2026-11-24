import 'dart:io';

import 'package:path/path.dart' show basename, dirname;

import '../utils.dart' show copyAssetFile;
import '../whisper_method_channel.dart';

/// Which Whisper model to run, and with which inference engine.
///
/// Each file is either a Flutter asset key (`assets/models/...`, copied once
/// to the app support directory) or an absolute path (e.g. a model downloaded
/// at first launch).
sealed class WhisperModel {
  const WhisperModel();

  /// Copy assets when needed and return the engine spec with absolute paths.
  Future<WhisperEngineSpec> resolve();
}

/// whisper.cpp with a ggml model (`ggml-base-q5_1.bin`, ...). Uses Metal on
/// iOS (and a bundled Core ML encoder when [useCoreMl]), the CPU on Android.
class GgmlWhisperModel extends WhisperModel {
  const GgmlWhisperModel(this.model, {this.useCoreMl = true});

  final String model;

  /// iOS: run the encoder with Core ML when `<name>-encoder.mlmodelc` is
  /// bundled next to the model (see [WhisperMethodChannel.prepareModel]).
  final bool useCoreMl;

  @override
  Future<WhisperEngineSpec> resolve() async {
    final path = await WhisperMethodChannel().prepareModel(
      model,
      useCoreMl: useCoreMl,
    );
    return WhisperEngineSpec.ggml(path);
  }
}

/// Whisper exported to ONNX and run by sherpa-onnx (ONNX Runtime), files as
/// published in the sherpa-onnx releases (`sherpa-onnx-whisper-base.tar.bz2`):
/// `base-encoder.onnx`, `base-decoder.onnx`, `base-tokens.txt`.
///
/// Measured on a Pixel 9a (CPU only): base fp32 transcribes a sentence in
/// ~0.4-1.5 s with an accuracy close to whisper.cpp base-q5_1, which takes
/// ~2 s. The `.int8.onnx` exports are faster but clearly less accurate.
///
/// Not supported by this engine: initial prompt, dynamic audio context,
/// temperature fallback, and the no-speech/log-probability stats used by
/// `HallucinationFilter` (its blocklist and repetition checks still apply).
class SherpaOnnxWhisperModel extends WhisperModel {
  const SherpaOnnxWhisperModel({
    required this.encoder,
    required this.decoder,
    required this.tokens,
    this.threads = 4,
  });

  final String encoder;
  final String decoder;
  final String tokens;

  /// ONNX Runtime intra-op threads (4 = the big cores of most phones).
  final int threads;

  @override
  Future<WhisperEngineSpec> resolve() async => WhisperEngineSpec.sherpaOnnx(
    encoder: await _sherpaFile(encoder),
    decoder: await _sherpaFile(decoder),
    tokens: await _sherpaFile(tokens),
    threads: threads,
  );
}

/// Asset key or absolute path → absolute path. Assets are copied once under
/// `models/sherpa/<asset folder>/<file>`: several NeMo models share file
/// names (`encoder.int8.onnx`...), so the folder keeps them apart.
Future<String> _sherpaFile(String source) async {
  if (source.startsWith('/')) {
    if (!await File(source).exists()) {
      throw ArgumentError.value(source, 'path', 'Model file not found');
    }
    return source;
  }
  return copyAssetFile(
    source,
    'models/sherpa/${basename(dirname(source))}/${basename(source)}',
  );
}

/// NVIDIA NeMo Canary (e.g. Canary-180M-Flash: en/de/es/fr) exported for
/// sherpa-onnx (`sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8`):
/// `encoder.int8.onnx`, `decoder.int8.onnx`, `tokens.txt`.
///
/// Measured on a real French voice (Pixel 9a, CPU): ~6.7% WER at ~0.26 s per
/// sentence, vs ~19% / ~1.5 s for whisper.cpp base-q8_0. ~200 MB, CC-BY-4.0
/// (credit NVIDIA in the app). The language must be explicit (no `auto`).
class SherpaOnnxCanaryModel extends WhisperModel {
  const SherpaOnnxCanaryModel({
    required this.encoder,
    required this.decoder,
    required this.tokens,
    this.threads = 4,
  });

  final String encoder;
  final String decoder;
  final String tokens;
  final int threads;

  @override
  Future<WhisperEngineSpec> resolve() async =>
      WhisperEngineSpec._(WhisperEngineSpec.canaryEngine, {
        'encoder': await _sherpaFile(encoder),
        'decoder': await _sherpaFile(decoder),
        'tokens': await _sherpaFile(tokens),
      }, threads);
}

/// NVIDIA NeMo transducer (Parakeet TDT, multilingual FastConformer...)
/// exported for sherpa-onnx: `encoder.int8.onnx`, `decoder.int8.onnx`,
/// `joiner.int8.onnx`, `tokens.txt`. Multilingual models detect the language
/// themselves.
///
/// Measured on a real French voice (Pixel 9a): FastConformer en/de/es/fr
/// (~130 MB) ~10.5% WER at ~0.07 s per sentence; Parakeet TDT 0.6B v3
/// (~670 MB) ~6.6% at ~0.19 s.
class SherpaOnnxTransducerModel extends WhisperModel {
  const SherpaOnnxTransducerModel({
    required this.encoder,
    required this.decoder,
    required this.joiner,
    required this.tokens,
    this.threads = 4,
  });

  final String encoder;
  final String decoder;
  final String joiner;
  final String tokens;
  final int threads;

  @override
  Future<WhisperEngineSpec> resolve() async =>
      WhisperEngineSpec._(WhisperEngineSpec.transducerEngine, {
        'encoder': await _sherpaFile(encoder),
        'decoder': await _sherpaFile(decoder),
        'joiner': await _sherpaFile(joiner),
        'tokens': await _sherpaFile(tokens),
      }, threads);
}

/// NVIDIA NeMo CTC model exported for sherpa-onnx: `model.int8.onnx`,
/// `tokens.txt`. Measured: FastConformer CTC en/de/es/fr (~126 MB) ~14.5% WER
/// at ~0.06 s per sentence on a Pixel 9a.
class SherpaOnnxNemoCtcModel extends WhisperModel {
  const SherpaOnnxNemoCtcModel({
    required this.model,
    required this.tokens,
    this.threads = 4,
  });

  final String model;
  final String tokens;
  final int threads;

  @override
  Future<WhisperEngineSpec> resolve() async => WhisperEngineSpec._(
    WhisperEngineSpec.nemoCtcEngine,
    {'model': await _sherpaFile(model), 'tokens': await _sherpaFile(tokens)},
    threads,
  );
}

/// Resolved engine description, sendable to the worker isolate.
class WhisperEngineSpec {
  const WhisperEngineSpec._(this.engine, this.files, this.threads);

  WhisperEngineSpec.ggml(String modelPath)
    : this._('ggml', {'model': modelPath}, 0);

  WhisperEngineSpec.sherpaOnnx({
    required String encoder,
    required String decoder,
    required String tokens,
    int threads = 4,
  }) : this._('sherpa-onnx', {
         'encoder': encoder,
         'decoder': decoder,
         'tokens': tokens,
       }, threads);

  static const String ggmlEngine = 'ggml';
  static const String sherpaWhisperEngine = 'sherpa-onnx';
  static const String canaryEngine = 'sherpa-canary';
  static const String transducerEngine = 'sherpa-transducer';
  static const String nemoCtcEngine = 'sherpa-nemo-ctc';

  /// One of [ggmlEngine], [sherpaWhisperEngine], [canaryEngine],
  /// [transducerEngine], [nemoCtcEngine].
  final String engine;
  final Map<String, String> files;
  final int threads;

  Map<String, Object> toMap() => {
    'engine': engine,
    'files': files,
    'threads': threads,
  };

  static WhisperEngineSpec fromMap(Map<dynamic, dynamic> map) =>
      WhisperEngineSpec._(
        map['engine'] as String,
        (map['files'] as Map).cast<String, String>(),
        map['threads'] as int,
      );

  @override
  String toString() => '$engine(${files.values.map(basename).join(', ')})';
}
