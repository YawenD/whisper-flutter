import 'dart:io';

import 'package:flutter/services.dart'
    show MethodChannel, PlatformException, rootBundle;
import 'package:path_provider/path_provider.dart';

class WhisperMethodChannel {
  static const MethodChannel _channel = MethodChannel('flutter_whisper_ggml');

  /// Copie le modèle depuis les assets si besoin
  Future<String> prepareModel(String existingModelPath) async {
    if (existingModelPath.startsWith('/')) {
      final file = File(existingModelPath);
      if (await file.exists()) {
        return existingModelPath;
      }
    }

    final data = await rootBundle.load(existingModelPath);
    final bytes = data.buffer.asUint8List();
    final dir = await getApplicationSupportDirectory();
    final outPath = '${dir.path}/model.bin';
    final outFile = File(outPath);

    if (!(await outFile.exists()) || (await outFile.length()) != bytes.length) {
      await outFile.create(recursive: true);
      await outFile.writeAsBytes(bytes, flush: true);
    }

    print('🧠 Modèle copié: $outPath');
    return outPath;
  }

  /// Transcrit un fichier WAV via Method Channel
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

  Future<String> transcribeStream({
    required String modelPath,
    required Stream<List<int>> audioStream,
  }) async {
    final result = await _channel.invokeMethod<String>('transcribeStream', {
      'modelPath': modelPath,
      'audioStream': audioStream,
    });

    if (result == null || result.isEmpty) {
      throw Exception('Erreur: transcription retournée vide');
    }

    return result;
  }
}
