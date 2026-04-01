import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart'
    show MethodChannel, PlatformException, rootBundle;
import 'package:path_provider/path_provider.dart';

class WhisperMethodChannel {
  static const MethodChannel _channel = MethodChannel('flutter_whisper_ggml');

  /// Copy the model from assets to disk if needed.
  Future<String> prepareModel(String existingModelPath) async {
    if (existingModelPath.startsWith('/')) {
      final file = File(existingModelPath);
      if (await file.exists()) {
        return existingModelPath;
      }
    }

    final dir = await getApplicationSupportDirectory();
    final outPath = '${dir.path}/model.bin';
    final outFile = File(outPath);

    // Check if file already exists BEFORE loading the asset into RAM.
    if (await outFile.exists()) {
      print('🧠 Model already present: $outPath');
      return outPath;
    }

    final data = await rootBundle.load(existingModelPath);
    final bytes = data.buffer.asUint8List();
    await outFile.create(recursive: true);
    await outFile.writeAsBytes(bytes, flush: true);

    print('🧠 Model copied: $outPath');
    return outPath;
  }

  /// Transcribe a WAV file via Method Channel.
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
        throw Exception('Erreur: transcription retournée vide');
      }

      return result;
    } on PlatformException catch (e) {
      throw Exception('Erreur de transcription: ${e.message}');
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
      throw Exception('Erreur: transcription retournée vide');
    }

    return result;
  }
}
