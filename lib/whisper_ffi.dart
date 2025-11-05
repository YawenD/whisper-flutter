import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

typedef _WhisperTranscribeFromFileNative =
    Pointer<Utf8> Function(
      Pointer<Utf8>, // model_path
      Pointer<Utf8>, // wav_path
    );
typedef _WhisperTranscribeFromFile =
    Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>);

typedef _CStringFreeNative = Void Function(Pointer<Utf8>);
typedef _CStringFree = void Function(Pointer<Utf8>);

class WhisperFFI {
  late final DynamicLibrary _lib;
  late final _WhisperTranscribeFromFile _transcribeFromFile;
  late final _CStringFree _freeCstr;

  WhisperFFI() {
    if (!Platform.isIOS) {
      throw UnsupportedError(
        'WhisperFFI is only supported on iOS. Use WhisperMethodChannel on Android.',
      );
    }
    _lib = DynamicLibrary.process();

    _transcribeFromFile = _lib
        .lookupFunction<
          _WhisperTranscribeFromFileNative,
          _WhisperTranscribeFromFile
        >('whisper_transcribe_from_file_bridge');

    _freeCstr = _lib.lookupFunction<_CStringFreeNative, _CStringFree>(
      'whisper_free_cstr_bridge',
    );
  }

  /// Copie le modèle depuis les assets si besoin
  Future<String> prepareModel(String existingModelPath) async {
    final maybeFile = File(existingModelPath);
    if (existingModelPath.startsWith('/') && await maybeFile.exists()) {
      return existingModelPath;
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

  /// Appel direct au C++ : lit et transcrit le fichier WAV côté natif
  Future<String> transcribe({
    required String modelPath,
    required String wavPath,
  }) async {
    final modelPtr = modelPath.toNativeUtf8();
    final wavPtr = wavPath.toNativeUtf8();

    final resPtr = _transcribeFromFile(modelPtr, wavPtr);
    malloc.free(modelPtr);
    malloc.free(wavPtr);

    if (resPtr.address == 0) {
      throw Exception('Erreur: transcription retournée vide');
    }

    final result = resPtr.toDartString();
    _freeCstr(resPtr);
    return result;
  }
}
