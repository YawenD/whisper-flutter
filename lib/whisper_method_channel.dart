import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint, kDebugMode;
import 'package:flutter/services.dart'
    show AssetManifest, MethodChannel, PlatformException, rootBundle;
import 'package:path/path.dart'
    show basename, basenameWithoutExtension, dirname, join;

import 'utils.dart';

class WhisperMethodChannel {
  static const MethodChannel _channel = MethodChannel('flutter_whisper_ggml');

  /// Return a file path for the model: an absolute path is used as is,
  /// anything else is treated as a Flutter asset and copied (atomically, once)
  /// to the app support directory under `models/<file name>`.
  ///
  /// iOS: when a Core ML encoder (`<name>-encoder.mlmodelc`, see
  /// [coreMlEncoderName]) is bundled in the same asset directory, it is copied
  /// next to the model, where whisper.cpp picks it up automatically to run the
  /// encoder on the Neural Engine. Pass [useCoreMl] false to remove a copied
  /// encoder and fall back to Metal (benchmarking).
  Future<String> prepareModel(
    String existingModelPath, {
    bool useCoreMl = true,
  }) async {
    if (existingModelPath.startsWith('/')) {
      if (await File(existingModelPath).exists()) {
        return existingModelPath;
      }
      throw ArgumentError.value(
        existingModelPath,
        'existingModelPath',
        'Model file not found',
      );
    }

    final outPath = await copyAssetFile(
      existingModelPath,
      'models/${basename(existingModelPath)}',
    );
    if (Platform.isIOS) {
      await _prepareCoreMlEncoder(existingModelPath, outPath, useCoreMl);
    }
    if (kDebugMode) {
      debugPrint('🧠 Model ready at: $outPath');
    }
    return outPath;
  }

  /// Name of the Core ML encoder whisper.cpp looks for next to [modelFile]:
  /// the extension and a trailing quantization suffix (`-q5_1`) are dropped,
  /// then `-encoder.mlmodelc` is appended (`ggml-base-q5_1.bin` →
  /// `ggml-base-encoder.mlmodelc`). Mirrors whisper_get_coreml_path_encoder.
  static String coreMlEncoderName(String modelFile) {
    var name = basenameWithoutExtension(modelFile);
    final dash = name.lastIndexOf('-');
    if (dash >= 0) {
      final suffix = name.substring(dash);
      if (suffix.length == 5 && suffix[1] == 'q' && suffix[3] == '_') {
        name = name.substring(0, dash);
      }
    }
    return '$name-encoder.mlmodelc';
  }

  Future<void> _prepareCoreMlEncoder(
    String modelAsset,
    String modelPath,
    bool useCoreMl,
  ) async {
    final encoder = coreMlEncoderName(modelAsset);
    final target = Directory(join(dirname(modelPath), encoder));
    if (!useCoreMl) {
      if (await target.exists()) {
        await target.delete(recursive: true);
      }
      return;
    }

    // A .mlmodelc is a directory: copy every bundled file under it.
    final prefix = '${dirname(modelAsset)}/$encoder/';
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final files = manifest.listAssets().where((a) => a.startsWith(prefix));
    for (final asset in files) {
      await copyAssetFile(
        asset,
        join('models', encoder, asset.substring(prefix.length)),
      );
    }
  }

  /// Transcribe a WAV file through the method channel.
  Future<String> transcribe({
    required String modelPath,
    required String wavPath,
  }) async {
    try {
      final result = await _channel.invokeMethod<String>('transcribe', {
        'filePath': wavPath,
        'modelPath': modelPath,
      });

      if (result == null || result.isEmpty) {
        throw Exception('Error: transcription result is empty');
      }

      return result;
    } on PlatformException catch (e) {
      throw Exception('Transcription error: ${e.message}');
    }
  }

  Future<String> transcribeData({
    required String modelPath,
    required Float32List audioData,
  }) async {
    final result = await _channel.invokeMethod<String>('transcribeData', {
      'modelPath': modelPath,
      'audioData': audioData,
    });
    if (result == null || result.isEmpty) {
      throw Exception('Error: transcription result is empty');
    }

    return result;
  }
}
