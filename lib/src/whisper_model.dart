import 'dart:io';

import 'package:path/path.dart' show basename;

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
  Future<WhisperEngineSpec> resolve() async {
    Future<String> file(String source) async {
      if (source.startsWith('/')) {
        if (!await File(source).exists()) {
          throw ArgumentError.value(source, 'path', 'Model file not found');
        }
        return source;
      }
      return copyAssetFile(source, 'models/sherpa/${basename(source)}');
    }

    return WhisperEngineSpec.sherpaOnnx(
      encoder: await file(encoder),
      decoder: await file(decoder),
      tokens: await file(tokens),
      threads: threads,
    );
  }
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
  }) : this._(
         'sherpa-onnx',
         {'encoder': encoder, 'decoder': decoder, 'tokens': tokens},
         threads,
       );

  /// `ggml` (whisper.cpp) or `sherpa-onnx`.
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
